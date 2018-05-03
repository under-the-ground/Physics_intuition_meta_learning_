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
#include "mat.h"

MATFile *pmat;
mxArray *pin;
mxArray *pout;
mxArray *pmeta;

#define TIMESTEP 0.001
#define TRAJLEN 1
#define NUMSAM 50
#define NUMTASK 10000
//#define INWIDTH 14
//#define OUTWIDTH 4

#define R_MIN 0.4
#define R_MAX 1.0
#define M_MIN 1.0
#define M_MAX 5.0
#define CR_MIN 0.1
#define CR_MAX 0.8
#define V_MIN 2.0
#define V_MAX 8.0
// keep restitution of first rect always 1.0

// m1, m2, cr
enum metav:int {m1, m2, cr, mLAST};
// q1(1), q2(3), dq1(2), dq2(2), pc(2), b1(2), b2(2)
enum inv:int {q1th, q2th, q2x, q2y, dq1w, dq1x, dq1y, dq2w, dq2x, dq2y, pcx, pcy, b1x, b1y, b2x, b2y, iLAST};
// dq1+(2), dq2+(2)
enum outv:int {ndq1w, ndq1x, ndq1y, ndq2w, ndq2x, ndq2y, oLAST};

const char *file = "training_rects_v2.mat";

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;


class MyWindow : public dart::gui::SimWindow
{
public:
    
    SkeletonPtr createRect(bool isFirst)
    {
        auto w = dart::math::random(R_MIN,R_MAX);
        auto h = dart::math::random(R_MIN,R_MAX);
        auto m = dart::math::random(M_MIN,M_MAX);
        double cr = isFirst? 1.0 : dart::math::random(CR_MIN,CR_MAX);
        SkeletonPtr rect_skel;
        if (isFirst)
        {
            rect_skel = Skeleton::create("rect_first");
            metaVec[metav::m1] = m;
            inVec[inv::b1x] = w;
            inVec[inv::b1y] = h;
        }
        else
        {
            rect_skel = Skeleton::create("rect_second");
            metaVec[metav::m2] = m;
            metaVec[metav::cr] = cr;
            inVec[inv::b2x] = w;
            inVec[inv::b2y] = h;
        }
        auto bn = rect_skel->createJointAndBodyNodePair<FreeJoint>().second;
        
//        std::shared_ptr<Shape> shape = std::make_shared<SphereShape>(r);
        std::shared_ptr<Shape> shape = std::make_shared<BoxShape>(Eigen::Vector3d(w, h, 0.36));
        
        auto shapeNode = bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(shape);
        shapeNode->getVisualAspect()->setColor(Eigen::Vector3d(1.0, 0.0, 0.0));
        
        // Set up inertia
        Inertia inertia;
        inertia.setMass(m);
        inertia.setMoment(shape->computeInertia(m));
        bn->setInertia(inertia);

        bn->setFrictionCoeff(0.0);
        bn->setRestitutionCoeff(cr);
        return rect_skel;
    }
    
    
    void setOrUpdatePosVel(SkeletonPtr& rect_skel, bool isFirst, double minDis)
    {
        Eigen::Vector6d pos;
        
        auto Pi = dart::math::constants<double>::pi();
        double theta = dart::math::random(-Pi,Pi);
        
//        if (isFirst) theta = 0.0; // ****
//        else theta = Pi/4;
        
        if (isFirst)
        {
            pos << 0, 0, theta, 0, 0, 0;
            rect_skel->getJoint(0)->setPositions(pos);
        }
        else
        {
            auto posVecAngle = dart::math::random(-Pi,Pi);
            
            posVecAngle = 0.0; // ****
            
            auto r = 0.5 * minDis;
            auto x = r * cos(posVecAngle);
            auto y = r * sin(posVecAngle);
            
            pos << 0, 0, theta, x, y, 0;
            rect_skel->getJoint(0)->setPositions(pos);
            
            // used FCL to calculate nearest points
            mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::FCLCollisionDetector::create());
            auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
            dart::collision::DistanceOption disOption;
            disOption.enableNearestPoints = true;
            dart::collision::DistanceResult disResult;
            auto dist = collisionGroup->distance(disOption, &disResult);
//            std::cout << dist << std::endl;
//            std::cout << disResult.nearestPoint1.transpose() << std::endl;
//            std::cout << disResult.nearestPoint2.transpose() << std::endl;
            if (dist > 0)
                pos.tail<3>() = pos.tail<3>() + (1.0000001) * (disResult.nearestPoint2 - disResult.nearestPoint1);
            else
                {pos[x] *= 3; pos[y] *= 3;}
            rect_skel->getJoint(0)->setPositions(pos);
            // switch back for contact force calculation
            mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
        }
        
        Eigen::Vector6d vel = Eigen::Vector6d::Zero();
        auto angv = dart::math::random(-V_MAX, V_MAX);
        angv = 0.0; // ****
        vel[2] = angv;
        
        auto vel_norm = dart::math::random(V_MIN, V_MAX);
        auto vel_angle = dart::math::random(-Pi, Pi);
        
        if (isFirst) vel_angle = 0;     // ****
        else vel_angle = Pi;
        
        vel[3] = vel_norm * cos(vel_angle);
        vel[4] = vel_norm * sin(vel_angle);
        
        // Create reference frames for setting the initial velocity
        Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
        centerTf.translation() = rect_skel->getCOM();
        SimpleFrame center(Frame::World(), "center", centerTf);
        Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], 0.0);
        Eigen::Vector3d w = vel[2] * Eigen::Vector3d::UnitZ();
        center.setClassicDerivatives(v, w);
        SimpleFrame ref(&center, "root_reference");
        // ?
        ref.setRelativeTransform(rect_skel->getBodyNode(0)->getTransform(&center));
        rect_skel->getJoint(0)->setVelocities(ref.getSpatialVelocity());
        
        if (isFirst)
        {
            inVec[inv::q1th] = pos[2];
            inVec[inv::dq1w] = vel[2];
            inVec[inv::dq1x] = vel[3];
            inVec[inv::dq1y] = vel[4];
        }
        else
        {
            inVec[inv::q2th] = pos[2];
            inVec[inv::q2x] = pos[3];
            inVec[inv::q2y] = pos[4];
            inVec[inv::dq2w] = vel[2];
            inVec[inv::dq2x] = vel[3];
            inVec[inv::dq2y] = vel[4];
        }
        
//        
//        VelIn = bNode->getSpatialVelocity(Frame::World(),Frame::World());
//        PosIn = mWorld->getSkeleton("hopper")->getPositions();
    }

    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);
        
        metaVec = Eigen::VectorXd::Zero(metav::mLAST);
        inVec = Eigen::VectorXd::Zero(inv::iLAST);
        outVec = Eigen::VectorXd::Zero(outv::oLAST);
        
        rect_first = createRect(true);
        rect_second = createRect(false);
        minDis = inVec[inv::b1x] + inVec[inv::b2x] + inVec[inv::b1y] + inVec[inv::b2y];
        
//        std::cout << minDis << std::endl;
        
        world->addSkeleton(rect_first);
        world->addSkeleton(rect_second);
        
        setOrUpdatePosVel(rect_first, true, minDis);
        setOrUpdatePosVel(rect_second, false, minDis);
        
        sampleCount = 0;
        ts = 0;
        
        N = 0;
        T = 0;
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
        if (ts == TRAJLEN && T < NUMTASK)
        {
            // check new vel
            auto ndq1 = rect_first->getBodyNode(0)->getSpatialVelocity(Frame::World(),Frame::World());
            auto ndq2 = rect_second->getBodyNode(0)->getSpatialVelocity(Frame::World(),Frame::World());
//            Eigen::Vector4d velchange(ndq1[3]-inVec[inv::dq1x], ndq1[4]-inVec[inv::dq1y], ndq2[3]-inVec[inv::dq2x], ndq2[4]-inVec[inv::dq2y]);
            Eigen::Vector3d f_con = Eigen::Vector3d::Zero();
            auto numCon = mWorld->getConstraintSolver()->getLastCollisionResult().getNumContacts();
            if (numCon > 0) f_con = mWorld->getConstraintSolver()->getLastCollisionResult().getContact(0).force;
            
            if (f_con.norm() < 1e-3)
            {
                std::cout << "discard: f_contact almost 0" << std::endl;
            }
            else
            {
                // finished running the traj for this sample
                
                // set outVec values
                outVec[outv::ndq1w] = ndq1[2];
                outVec[outv::ndq1x] = ndq1[3];
                outVec[outv::ndq1y] = ndq1[4];
                
                outVec[outv::ndq2w] = ndq2[2];
                outVec[outv::ndq2x] = ndq2[3];
                outVec[outv::ndq2y] = ndq2[4];
                
                // store
                storeOneInFile();
                
                std::cout << "S" << N << ": " << metaVec.transpose() << " |" << inVec.transpose() << " |" << outVec.transpose() <<std::endl;
                N++;
                
                if (N == NUMSAM)
                {
                    // finished sample for this task, start sampling for next task
                    std::cout << "finished task " << T << std::endl;
                    
                    mWorld->removeSkeleton(rect_first);
                    mWorld->removeSkeleton(rect_second);
                    
                    // update M, Cr, and R's
                    rect_first = createRect(true);
                    rect_second = createRect(false);
                    minDis = inVec[inv::b1x] + inVec[inv::b2x] + inVec[inv::b1y] + inVec[inv::b2y];
                    mWorld->addSkeleton(rect_first);
                    mWorld->addSkeleton(rect_second);

                    T++;
                    N = 0;
                }
            }
            
            // reset q and dp etc
            setOrUpdatePosVel(rect_first, true, minDis);
            setOrUpdatePosVel(rect_second, false, minDis);
            
            ts = 0;
        }
        
        // check collision, if collision, update pc
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        
//        if (collision && result.getNumContacts() == 2)
        if (collision)
        {
            inVec[inv::pcx] = result.getContact(0).point[0];
            inVec[inv::pcy] = result.getContact(0).point[1];
            
            // run simulation
//            SimWindow::timeStepping();
            
//            mWorld->getConstraintSolver()->solve();

        }
        
//Integrate velocity for unconstrained skeletons
//        for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
//        {
//            auto skel = mWorld->getSkeleton(i);
//            if (!skel->isMobile())
//                continue;
//            skel->computeForwardDynamics();
//            skel->integrateVelocities(mWorld->getTimeStep());
//        }
//        mWorld->getConstraintSolver()->solve();
//        for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
//        {
//            auto skel = mWorld->getSkeleton(i);
//            if (!skel->isMobile())
//                continue;
//            
//            if (skel->isImpulseApplied())
//            {
//                skel->computeImpulseForwardDynamics();
//                skel->setImpulseApplied(false);
//            }
//            
//            skel->integratePositions(mWorld->getTimeStep());
//        }

//        else
//        {
//            if (T < NUMTASK)
//            {
//                auto num = result.getNumContacts();
//                std::cout << "discard: no collision" << std::endl;
//                for (int j=0; j<num; j++)
//                {
//                    std::cout << result.getContact(j).point.transpose() << std::endl;
//                }
//            }
//            
//        }
        SimWindow::timeStepping();
        ts++;
    }
    
    void storeOneInFile()
    {
        if (sampleCount < NUMSAM*NUMTASK)
        {
            memcpy((void *)(mxGetPr(pin) + sampleCount*inv::iLAST), (void *)(inVec.data()), sizeof(double)*inv::iLAST);
            memcpy((void *)(mxGetPr(pout) + sampleCount*outv::oLAST), (void *)(outVec.data()), sizeof(double)*outv::oLAST);
            memcpy((void *)(mxGetPr(pmeta) + sampleCount*metav::mLAST), (void *)(metaVec.data()), sizeof(double)*metav::mLAST);
            
            sampleCount++;
        }
        if (sampleCount == NUMSAM*NUMTASK)
        {
            int status = matPutVariable(pmat, "In", pin);
            if (status != 0) {
                printf("Error using matPutVariable in\n");
                exit(1);
            }
            
            mxDestroyArray(pin);
            
            status = matPutVariable(pmat, "Out", pout);
            if (status != 0) {
                printf("Error using matPutVariable out\n");
                exit(1);
            }
            
            mxDestroyArray(pout);
            
            status = matPutVariable(pmat, "Meta", pmeta);
            if (status != 0) {
                printf("Error using matPutVariable meta\n");
                exit(1);
            }
            
            mxDestroyArray(pmeta);
            
            if (matClose(pmat) != 0) {
                printf("Error closing file %s\n",file);
                exit(1);
            }
            printf("Done\n");
            
            sampleCount++;
        }
    }
    
    int ts;
    int sampleCount;
    
    int N;
    int T;
    
    double minDis;
    
    SkeletonPtr rect_first;
    SkeletonPtr rect_second;
    
    Eigen::VectorXd metaVec;
    Eigen::VectorXd inVec;
    Eigen::VectorXd outVec;
    
protected:
};



int main(int argc, char* argv[])
{
    
    WorldPtr world = std::make_shared<World>();
    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));

    world->setTimeStep(TIMESTEP);
    world->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
    
//    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/skel/mytest/hopperfoot2d.skel");
//    assert(world != nullptr);
    
    MyWindow window(world);
    
    std::cout << "space bar: simulation on/off" << std::endl;
    
    printf("Creating file %s...\n\n", file);
    pmat = matOpen(file, "w");
    if (pmat == NULL) {
        printf("Error creating file %s\n", file);
        exit(1);
    }
    
    pin = mxCreateDoubleMatrix(inv::iLAST,NUMSAM*NUMTASK,mxREAL);
    if (pin == NULL) {
        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
        printf("Unable to create mxArray.\n");
        exit(1);
    }
    
    pout = mxCreateDoubleMatrix(outv::oLAST,NUMSAM*NUMTASK,mxREAL);
    if (pout == NULL) {
        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
        printf("Unable to create mxArray.\n");
        exit(1);
    }
    
    pmeta = mxCreateDoubleMatrix(metav::mLAST,NUMSAM*NUMTASK,mxREAL);
    if (pmeta == NULL) {
        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
        printf("Unable to create mxArray.\n");
        exit(1);
    }
    
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
