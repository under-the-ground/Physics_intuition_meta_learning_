/*
 * Copyright (c) 2015-2016, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2015-2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/ode/ode.hpp"
#include <dart/utils/utils.hpp>
#include <math.h>

#include "tiny_dnn/tiny_dnn.h"

// q1(1), q2(3), dq1(2), dq2(2), pc(2), b1(2), b2(2)
enum inv:int {q1th, q2th, q2x, q2y, dq1x, dq1y, dq2x, dq2y, pcx, pcy, b1x, b1y, b2x, b2y, iLAST};


// 2.23680119402707	2.40897893905640	0.224081955850124	0.421516253799200	0.421516253799200	0.391767769120634	0.391767769120634
//1.80645945295692	3.88848821446300	0.593752539809793	0.488866581954062	0.488866581954062	0.316403215751052	0.316403215751052
#define TIMESTEP 0.001

#define R1 0.488866581954062
#define R2 0.316403215751052
#define M1 1.80645945295692
#define M2 3.88848821446300
#define CR 0.593752539809793

#define USEDART false

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

class MyWindow : public dart::gui::SimWindow
{
public:
    
    SkeletonPtr createBall(bool isFirst)
    {
        auto r = isFirst? R1 : R2;
        auto m = isFirst? M1 : M2;
        double cr = isFirst? 1.0 : CR;
        SkeletonPtr ball_skel;
        if (isFirst)
        {
            ball_skel = Skeleton::create("ball_first");
            inVec[inv::b1x] = inVec[inv::b1y] = r;
        }
        else
        {
            ball_skel = Skeleton::create("ball_second");
            inVec[inv::b2x] = inVec[inv::b2y] = r;
        }
        auto bn = ball_skel->createJointAndBodyNodePair<FreeJoint>().second;
        
        std::shared_ptr<Shape> shape = std::make_shared<SphereShape>(r);
        //        shape = std::make_shared<BoxShape>(Eigen::Vector3d(0.08, 0.08, 0.08));
        
        auto shapeNode = bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(shape);
        shapeNode->getVisualAspect()->setColor(Eigen::Vector3d(1.0, 0.0, 0.0));
        
        // Set up inertia
        Inertia inertia;
        inertia.setMass(m);
        inertia.setMoment(shape->computeInertia(m));
        bn->setInertia(inertia);
        
        bn->setFrictionCoeff(0.0);
        bn->setRestitutionCoeff(cr);
        return ball_skel;
    }
    
    void setVel(SkeletonPtr& ball_skel, Eigen::Vector6d vel)
    {
        // Create reference frames for setting the initial velocity
        Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
        centerTf.translation() = ball_skel->getCOM();
        SimpleFrame center(Frame::World(), "center", centerTf);
        Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], 0.0);
        Eigen::Vector3d w = vel[2] * Eigen::Vector3d::UnitZ();
        center.setClassicDerivatives(v, w);
        SimpleFrame ref(&center, "root_reference");
        // ?
        ref.setRelativeTransform(ball_skel->getBodyNode(0)->getTransform(&center));
        ball_skel->getJoint(0)->setVelocities(ref.getSpatialVelocity());
    }

    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);
        
//        metaVec = Eigen::VectorXd::Zero(metav::mLAST);
        inVec = Eigen::VectorXd::Zero(inv::iLAST);
//        outVec = Eigen::VectorXd::Zero(outv::oLAST); 
        
        ball_first = createBall(true);
        ball_second = createBall(false);
        
        world->addSkeleton(ball_first);
        world->addSkeleton(ball_second);
        
        Eigen::Vector6d pos = Eigen::Vector6d::Zero();
        Eigen::Vector6d vel = Eigen::Vector6d::Zero();
        vel[3] = 5;
//        vel[4] = -2;
        ball_first->getJoint(0)->setPositions(pos); // note the pos is not the pos fed into the NN
        setVel(ball_first, vel);
        
        pos = Eigen::Vector6d::Zero();
        vel = Eigen::Vector6d::Zero();
        pos[3] = 2;
//        pos[4] = -2;
        vel[3] = -5;
//        vel[4] = 3;
        ball_second->getJoint(0)->setPositions(pos);
        setVel(ball_second, vel);

        if (!USEDART) net.load("maml/net-maml_2cir_181389059-may2-0035");
        
        flag = false;
    }
    
    void keyboard(unsigned char key, int x, int y) override
    {
        switch(key)
        {
            default:
                SimWindow::keyboard(key, x, y);
        }
    }
    
    void drawWorld() const override
    {
        // Make sure lighting is turned on and that polygons get filled in
        glEnable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        SimWindow::drawWorld();
    }
    
    void timeStepping() override
    {
//        // Integrate velocity for unconstrained skeletons
//        for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
//        {
//            auto skel = mWorld->getSkeleton(i);
//            if (!skel->isMobile())
//                continue;
//            
//            skel->computeForwardDynamics();
//            skel->integrateVelocities(mWorld->getTimeStep());
//        }
        
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        
        if (collision)
        {
            if (USEDART || flag)
            {
                mWorld->getConstraintSolver()->solve();
            }
            else
            {
                auto q1in = ball_first->getPositions();
                auto q2in = ball_second->getPositions();
                inVec[inv::q1th] = q1in[2];
                inVec[inv::q2th] = q2in[2];
                inVec[inv::q2x] = q2in[3] - q1in[3];   // the NN assume first ball to be at (0,0)
                inVec[inv::q2y] = q2in[4] - q1in[4];
    
                auto dq1in = ball_first->getBodyNode(0)->getSpatialVelocity(Frame::World(),Frame::World());
                auto dq2in = ball_second->getBodyNode(0)->getSpatialVelocity(Frame::World(),Frame::World());
                inVec[inv::dq1x] = dq1in[3];
                inVec[inv::dq1y] = dq1in[4];
                inVec[inv::dq2x] = dq2in[3];
                inVec[inv::dq2y] = dq2in[4];
    
                inVec[inv::pcx] = result.getContact(0).point[0] - q1in[3];    // asuume only 1 contact point, the NN assume first ball to be at (0,0)
                inVec[inv::pcy] = result.getContact(0).point[1] - q1in[4];
    
                std::cout << inVec.transpose() << std::endl;
    
                // run NN
                vec_t input_nn;
                input_nn.assign(inVec.data(), inVec.data()+inv::iLAST);
                vec_t output_nn = net.predict(input_nn);
                
//                std::cout << output_nn << std::endl;
    
                Eigen::Vector6d vel;
                vel << 0,0,0, *(output_nn.begin()), *(output_nn.begin()+1),0;
                std::cout << vel.transpose() << std::endl;
//                vel << 0,0,0, -1, 0,0;
                setVel(ball_first, vel);
                vel << 0,0,0, *(output_nn.begin()+2), *(output_nn.begin()+3),0;
                std::cout << vel.transpose() << std::endl;
//                vel << 0,0,0, 1, 0,0;
                setVel(ball_second, vel);
                
                flag = true;
            }

            
        }
        
        if (USEDART || flag)
        {
            // modified/trimmed original time stepping
            SimWindow::timeStepping();
        }
        else
        {
            
            for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
            {
                auto skel = mWorld->getSkeleton(i);
                skel->integratePositions(mWorld->getTimeStep());
            }
        }
    }
    
    SkeletonPtr ball_first;
    SkeletonPtr ball_second;
    
    Eigen::VectorXd inVec;
    
    network<sequential> net;
    
    bool flag;
protected:
};


int main(int argc, char* argv[])
{
    WorldPtr world = std::make_shared<World>();
    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
    
    world->setTimeStep(TIMESTEP);
    world->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
    
    MyWindow window(world);
    
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
