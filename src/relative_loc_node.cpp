#include <string>
#include <map>
#include <deque>
#include <thread>
#include <mutex>
#include <eigen3/Eigen/Eigen>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include "nlink_parser/LinktrackNodeframe3.h"
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose.h>
#include "relative_loc/Drift.h"
#include "relative_loc/DistanceMeas.h"
#include "relative_loc/AnonymousBearingMeas.h"
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>

using namespace std;
using namespace Eigen;

// Measurements
struct Jupiter
{
    int id;
    Vector3d position;
    Quaterniond orientation;
};
struct Ganymede
{
    int id;
    Vector3d position;
    Quaterniond orientation;
};
struct DistMeas
{
    ros::Time timestamp;
    Jupiter jupiter;
    Ganymede ganymede;
    double distance;
};
struct AnonymousBearingMeas
{
    ros::Time timestamp;
    Jupiter jupiter;
    Vector3d bearing;
};
struct BearingMeas
{
    ros::Time timestamp;
    Jupiter jupiter;
    Ganymede ganymede;
    Vector3d bearing;
};

// Decision variable
struct Drift
{
    Vector3d translation = Vector3d::Zero();
    double yaw = 0.0;
};

// Cost functions
struct RegularizationCostFunctor
{
    RegularizationCostFunctor(const Vector3d &initial_translation, double initial_yaw, double weight, bool optimize_z)
        : initial_translation_(initial_translation), initial_yaw_(initial_yaw), weight_(weight), optimize_z_(optimize_z) {}

    template <typename T>
    bool operator()(const T *const translation, const T *const yaw, T *residuals) const
    {
        residuals[0] = T(weight_) * (translation[0] - T(initial_translation_(0)));
        residuals[1] = T(weight_) * (translation[1] - T(initial_translation_(1)));
        if (optimize_z_)
            residuals[2] = T(weight_) * (translation[2] - T(initial_translation_(2)));
        else
            residuals[2] = T(0.0);
        residuals[3] = T(weight_) * (*yaw - T(initial_yaw_));
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d &initial_translation, double initial_yaw, double weight, bool optimize_z)
    {
        return (new ceres::AutoDiffCostFunction<RegularizationCostFunctor, 4, 3, 1>(
            new RegularizationCostFunctor(initial_translation, initial_yaw, weight, optimize_z)));
    }

private:
    const Vector3d initial_translation_;
    const double initial_yaw_;
    const double weight_;
    const bool optimize_z_;
};
struct DistanceCostFunctor
{
    DistanceCostFunctor(const Vector3d &jupiter_odom_p, const Vector3d &ganymede_odom_p,
                        double distance, double weight, bool optimize_z)
        : jupiter_odom_p_(jupiter_odom_p), ganymede_odom_p_(ganymede_odom_p), distance_(distance), weight_(weight), optimize_z_(optimize_z) {}

    template <typename T>
    bool operator()(const T *const jupiter_drift_p, const T *const jupiter_drift_yaw,
                    const T *const ganymede_drift_p, const T *const ganymede_drift_yaw, T *residuals) const
    {
        // Compute drift quaternion and translation for Jupiter
        Map<const Matrix<T, 3, 1>> jupiter_drift_p_(jupiter_drift_p);
        Quaternion<T> jupiter_drift_q = Quaternion<T>(AngleAxis<T>(*jupiter_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Jupiter's odometry position
        Matrix<T, 3, 1> jupiter_p = jupiter_drift_q * jupiter_odom_p_.cast<T>() + jupiter_drift_p_;

        // Compute drift quaternion and translation for Ganymede
        Map<const Matrix<T, 3, 1>> ganymede_drift_p_(ganymede_drift_p);
        Quaternion<T> ganymede_drift_q = Quaternion<T>(AngleAxis<T>(*ganymede_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Ganymede's odometry position
        Matrix<T, 3, 1> ganymede_p = ganymede_drift_q * ganymede_odom_p_.cast<T>() + ganymede_drift_p_;

        // FIXME: the correct premise here is that drift_q is only rotating in the yaw direction
        if (!optimize_z_)
        {
            jupiter_p(2) = T(jupiter_odom_p_(2));
            ganymede_p(2) = T(ganymede_odom_p_(2));
        }

        Matrix<T, 3, 1> relative_p = ganymede_p - jupiter_p;

        residuals[0] = T(weight_) * (T(distance_) - relative_p.norm());

        return true;
    }

    template <typename T>
    bool operator()(const T *const jupiter_drift_p, const T *const jupiter_drift_yaw, T *residuals) const
    {
        // Compute drift quaternion and translation for Jupiter
        Map<const Matrix<T, 3, 1>> jupiter_drift_p_(jupiter_drift_p);
        Quaternion<T> jupiter_drift_q = Quaternion<T>(AngleAxis<T>(*jupiter_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Jupiter's odometry position
        Matrix<T, 3, 1> jupiter_p = jupiter_drift_q * jupiter_odom_p_.cast<T>() + jupiter_drift_p_;

        // FIXME: the correct premise here is that drift_q is only rotating in the yaw direction
        if (!optimize_z_)
        {
            jupiter_p(2) = T(jupiter_odom_p_(2));
        }

        Matrix<T, 3, 1> relative_p = ganymede_odom_p_.cast<T>() - jupiter_p;

        residuals[0] = T(weight_) * (T(distance_) - relative_p.norm());

        return true;
    }

    static ceres::CostFunction *Create(const Vector3d &jupiter_odom_p, const Vector3d &ganymede_odom_p,
                                       double distance, double weight, bool optimize_z)
    {
        return (new ceres::AutoDiffCostFunction<DistanceCostFunctor, 1, 3, 1, 3, 1>(
            new DistanceCostFunctor(jupiter_odom_p, ganymede_odom_p, distance, weight, optimize_z)));
    }

    static ceres::CostFunction *CreateAnchorCost(const Vector3d &jupiter_odom_p, const Vector3d &ganymede_odom_p,
                                                 double distance, double weight, bool optimize_z)
    {
        return (new ceres::AutoDiffCostFunction<DistanceCostFunctor, 1, 3, 1>(
            new DistanceCostFunctor(jupiter_odom_p, ganymede_odom_p, distance, weight, optimize_z)));
    }

private:
    const Vector3d jupiter_odom_p_;
    const Vector3d ganymede_odom_p_;
    const double distance_;
    const double weight_;
    const bool optimize_z_;
};
struct BearingCostFunctor
{
    BearingCostFunctor(const Vector3d &jupiter_odom_p, const Quaterniond &jupiter_odom_q,
                       const Vector3d &ganymede_odom_p, const Quaterniond &ganymede_odom_q,
                       const Vector3d &bearing, double weight, bool optimize_z)
        : jupiter_odom_p_(jupiter_odom_p), jupiter_odom_q_(jupiter_odom_q), ganymede_odom_p_(ganymede_odom_p), ganymede_odom_q_(ganymede_odom_q), bearing_(bearing), weight_(weight), optimize_z_(optimize_z) {}

    template <typename T>
    bool operator()(const T *const jupiter_drift_p, const T *const jupiter_drift_yaw,
                    const T *const ganymede_drift_p, const T *const ganymede_drift_yaw, T *residuals) const
    {
        // Compute drift quaternion and translation for Jupiter
        Map<const Matrix<T, 3, 1>> jupiter_drift_p_(jupiter_drift_p);
        Quaternion<T> jupiter_drift_q = Quaternion<T>(AngleAxis<T>(*jupiter_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Jupiter's odometry
        Matrix<T, 3, 1> jupiter_p = jupiter_drift_q * jupiter_odom_p_.cast<T>() + jupiter_drift_p_;
        Quaternion<T> jupiter_q = jupiter_drift_q * jupiter_odom_q_.cast<T>();

        // Compute drift quaternion and translation for Ganymede
        Map<const Matrix<T, 3, 1>> ganymede_drift_p_(ganymede_drift_p);
        Quaternion<T> ganymede_drift_q = Quaternion<T>(AngleAxis<T>(*ganymede_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Ganymede's odometry position
        Matrix<T, 3, 1> ganymede_p = ganymede_drift_q * ganymede_odom_p_.cast<T>() + ganymede_drift_p_;

        // FIXME: the correct premise here is that drift_q is only rotating in the yaw direction
        if (!optimize_z_)
        {
            jupiter_p(2) = T(jupiter_odom_p_(2));
            ganymede_p(2) = T(ganymede_odom_p_(2));
        }

        Matrix<T, 3, 1> bearing_from_odom = (jupiter_q.inverse() * (ganymede_p - jupiter_p)).normalized();

        residuals[0] = T(weight_) * (T(1.0) - bearing_from_odom.dot(bearing_.cast<T>()));

        return true;
    }

    static ceres::CostFunction *Create(const Vector3d &jupiter_odom_p, const Quaterniond &jupiter_odom_q,
                                       const Vector3d &ganymede_odom_p, const Quaterniond &ganymede_odom_q,
                                       const Vector3d &bearing, double weight, bool optimize_z)
    {
        return (new ceres::AutoDiffCostFunction<BearingCostFunctor, 1, 3, 1, 3, 1>(
            new BearingCostFunctor(jupiter_odom_p, jupiter_odom_q, ganymede_odom_p, ganymede_odom_q, bearing, weight, optimize_z)));
    }

private:
    const Vector3d jupiter_odom_p_;
    const Quaterniond jupiter_odom_q_;
    const Vector3d ganymede_odom_p_;
    const Quaterniond ganymede_odom_q_; // FIXME: unused actually
    const Vector3d bearing_;
    const double weight_;
    const bool optimize_z_;
};

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;

    ros::init(argc, argv, "relative_loc_node");
    ros::NodeHandle nh("~");

    string hyperparam_path = argv[1];
    cv::FileStorage param(hyperparam_path, cv::FileStorage::READ);

    int glog_severity_level = param["glog_severity_level"];
    switch (glog_severity_level)
    {
    case 0:
        FLAGS_stderrthreshold = google::INFO;
        break;
    case 1:
        FLAGS_stderrthreshold = google::WARNING;
        break;
    case 2:
        FLAGS_stderrthreshold = google::ERROR;
        break;
    case 3:
        FLAGS_stderrthreshold = google::FATAL;
        break;
    default:
        break;
    }

    CHECK(getenv("DRONE_ID") != nullptr);
    int self_id = atoi(getenv("DRONE_ID"));

    Vector3d drift_p(0.0, 0.0, 0.0);
    Quaterniond drift_q(1.0, 0.0, 0.0, 0.0);
    mutex drift_mtx;
    auto applyDrift = [](const Vector3d &drift_p, const Quaterniond &drift_q,
                         Vector3d &odom_p, Quaterniond &odom_q)
    {
        odom_p = drift_q * odom_p + drift_p;
        odom_q = drift_q * odom_q;
    };
    ros::Publisher revised_odom_pub = nh.advertise<nav_msgs::Odometry>((string)param["revised_odom_topic"], 1000);
    auto odomCallback = [&](const nav_msgs::Odometry::ConstPtr &msg)
    {
        Vector3d revised_odom_p(msg->pose.pose.position.x,
                                msg->pose.pose.position.y,
                                msg->pose.pose.position.z);
        Quaterniond revised_odom_q(msg->pose.pose.orientation.w,
                                   msg->pose.pose.orientation.x,
                                   msg->pose.pose.orientation.y,
                                   msg->pose.pose.orientation.z);

        {
            lock_guard<mutex> lock(drift_mtx);
            applyDrift(drift_p, drift_q, revised_odom_p, revised_odom_q);
        }

        nav_msgs::Odometry revised_odom_msg(*msg);
        revised_odom_msg.pose.pose.position.x = revised_odom_p.x();
        revised_odom_msg.pose.pose.position.y = revised_odom_p.y();
        revised_odom_msg.pose.pose.position.z = revised_odom_p.z();
        revised_odom_msg.pose.pose.orientation.w = revised_odom_q.w();
        revised_odom_msg.pose.pose.orientation.x = revised_odom_q.x();
        revised_odom_msg.pose.pose.orientation.y = revised_odom_q.y();
        revised_odom_msg.pose.pose.orientation.z = revised_odom_q.z();

        revised_odom_pub.publish(revised_odom_msg);
    };
    ros::Subscriber odom_sub =
        nh.subscribe<nav_msgs::Odometry>((string)param["odom_topic"], 1000,
                                         odomCallback,
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());

    const char *is_center = getenv("IS_REL_LOC_CENTER");
    if (is_center != nullptr &&
        (string(is_center) == "true" ||
         string(is_center) == "True" ||
         string(is_center) == "TRUE" ||
         string(is_center) == "1"))
    {
        LOG(INFO) << "\033[32m============= This is the central node of the relative localization system *~* =============\033[0m";

            set<int> drone_id;
            cv::FileNode id_node = param["drone_id"];
            for (auto it = id_node.begin(); it != id_node.end(); ++it)
            {
                int id;
                *it >> id;
                CHECK((drone_id.insert(id)).second);
            }
            CHECK(drone_id.find(self_id) != drone_id.end());
            int drone_num = drone_id.size();

            set<int> uwb_anchor_id;
            map<int, Vector3d> uwb_anchor_position;
            cv::FileNode uwb_node = param["uwb_anchor_id_and_position"];
            for (auto it = uwb_node.begin(); it != uwb_node.end(); ++it)
            {
                int id = (*it)["id"];
                CHECK((uwb_anchor_id.insert(id)).second);
                std::vector<double> pos;
                (*it)["position"] >> pos;
                CHECK(pos.size() == 3) << "We live in a 3 dimensional world, homie ^^";
                uwb_anchor_position[id] = Vector3d(pos[0], pos[1], pos[2]);
            }

        auto checkNoDuplicateID = [](const set<int> &drone_id, const set<int> &uwb_anchor_id)
        {
            set<int> union_;
            set_union(drone_id.begin(), drone_id.end(),
                      uwb_anchor_id.begin(), uwb_anchor_id.end(),
                      inserter(union_, union_.begin()));
            return union_.size() == drone_id.size() + uwb_anchor_id.size();
        };
        CHECK(checkNoDuplicateID(drone_id, uwb_anchor_id));

        mutex swarm_odom_mtx,
            dist_meas_raw_mtx,
            dist_meas_mtx,
            anonymous_bearing_meas_raw_mtx,
            anonymous_bearing_meas_mtx,
            bearing_meas_mtx;

        map<int, deque<nav_msgs::Odometry>> swarm_odom_raw;
        auto swarmOdomCallback = [&](const nav_msgs::Odometry::ConstPtr &msg)
        {
            CHECK_EQ(msg->child_frame_id.substr(0, 6), "drone_");

            int id = atoi(msg->child_frame_id.substr(6, 10).c_str());
            CHECK(drone_id.find(id) != drone_id.end()) << "Drone " << id << " is not in the swarm #^#";
            if (abs((msg->header.stamp - ros::Time::now()).toSec()) > 1.0)
            {
                LOG(WARNING) << "Timestamp of a odom from drone " << id << " is more than 1.0s later (or earlier) than current time @_@";
            }

            lock_guard<mutex> lock(swarm_odom_mtx);

            swarm_odom_raw[id].push_back(*msg);

            if (swarm_odom_raw[id].size() > 10 * 3.0 * drone_num) // 10 seconds buffer FIXME: make source freq configurable
            {
                swarm_odom_raw[id].pop_front();
            }
        };
        ros::Subscriber swarm_odom_sub =
            nh.subscribe<nav_msgs::Odometry>((string)param["swarm_odom_topic"], 1000,
                                             swarmOdomCallback,
                                             ros::VoidConstPtr(),
                                             ros::TransportHints().tcpNoDelay());

        map<int, deque<relative_loc::DistanceMeas>> swarm_dist_meas_raw;
        auto distanceCallback = [&](const relative_loc::DistanceMeas::ConstPtr &msg)
        {
            int id = msg->distance_meas.id;
            CHECK(drone_id.find(id) != drone_id.end()) << "Drone " << id << " is not in the swarm #^#";
            CHECK(uwb_anchor_id.find(id) == uwb_anchor_id.end()) << "Drone " << id << " is a fxxking uwb anchor #^#";
            if (abs((msg->header.stamp - ros::Time::now()).toSec()) > 1.0)
            {
                LOG(WARNING) << "Timestamp of a distance measurement from drone " << id << " is more than 1.0s later (or earlier) than current time @_@";
            }

            lock_guard<mutex> lock(dist_meas_raw_mtx);

            swarm_dist_meas_raw[id].push_back(*msg);

            if (swarm_dist_meas_raw[id].size() > 10 * 10.0 * drone_num) // 10 seconds buffer FIXME: make source freq configurable
            {
                swarm_dist_meas_raw[id].pop_front();
            }
        };
        ros::Subscriber distance_sub =
            nh.subscribe<relative_loc::DistanceMeas>((string)param["distance_measurement_topic"], 1000,
                                                     distanceCallback,
                                                     ros::VoidConstPtr(),
                                                     ros::TransportHints().tcpNoDelay());

        map<int, deque<relative_loc::AnonymousBearingMeas>> swarm_anonymous_bearing_meas_raw;
        auto bearingCallback = [&](const relative_loc::AnonymousBearingMeas::ConstPtr &msg)
        {
            CHECK(drone_id.find(msg->id) != drone_id.end()) << "Drone " << msg->id << " is not in the swarm #^#";
            if (abs((msg->anonymous_bearing.header.stamp - ros::Time::now()).toSec()) > 1.0)
            {
                LOG(WARNING) << "Timestamp of a bearing measurement from drone " << msg->id << " is more than 1.0s later (or earlier) than current time @_@";
            }

            lock_guard<mutex> lock(anonymous_bearing_meas_raw_mtx);

            swarm_anonymous_bearing_meas_raw[msg->id].push_back(*msg);

            if (swarm_anonymous_bearing_meas_raw[msg->id].size() > 10 * 5.0 * drone_num) // 10 seconds buffer FIXME: make source freq configurable
            {
                swarm_anonymous_bearing_meas_raw[msg->id].pop_front();
            }
        };
        ros::Subscriber bearing_sub =
            nh.subscribe<relative_loc::AnonymousBearingMeas>((string)param["bearing_measurement_topic"], 1000,
                                                             bearingCallback,
                                                             ros::VoidConstPtr(),
                                                             ros::TransportHints().tcpNoDelay());

        ros::AsyncSpinner spinner(4);
        spinner.start();

        deque<DistMeas> swarm_dist_meas;
        deque<AnonymousBearingMeas> swarm_anonymous_bearing_meas;
        atomic<bool> fresh_dist_meas_ready{false};
        atomic<bool> fresh_anonymous_bearing_meas_ready{false};
        auto alignMeasAndOdomByTimestamp = [&](const int id, const ros::Time &meas_timestamp, geometry_msgs::Pose &pose_from_odom)
        {
            lock_guard<mutex> lock(swarm_odom_mtx);

            const deque<nav_msgs::Odometry> &odoms = swarm_odom_raw[id];
            if (odoms.empty())
            {
                LOG(ERROR) << "No odom received for drone " << id << " #^#";
                return false;
            }

            const double tolerance = 0.33; // Period of receiving other drone's odom TODO: make this configurable
            if (meas_timestamp < odoms.front().header.stamp - ros::Duration(tolerance) ||
                meas_timestamp > odoms.back().header.stamp + ros::Duration(tolerance))
            {
                LOG(ERROR) << ((meas_timestamp > odoms.back().header.stamp + ros::Duration(tolerance)) ? "Distance or bearing measurement timestamp >> lastest odom timestamp #^#" : "Distance or bearing measurement timestamp << earliest odom timestamp #^#");
                return false;
            }

            // Find the closest odom by iterating from the end
            auto closest_it = odoms.rbegin();
            double closest_diff = numeric_limits<double>::infinity();

            for (auto it = odoms.rbegin(); it != odoms.rend(); ++it)
            {
                double diff = abs((meas_timestamp - it->header.stamp).toSec());
                if (diff < closest_diff)
                {
                    closest_diff = diff;
                    closest_it = it;
                }
                else
                {
                    // Since the deque is ordered, we can break early if the difference starts increasing
                    break;
                }
            }

            pose_from_odom = (*closest_it).pose.pose;
            return true;
        };
        thread align_meas_and_odom_by_timestamp_thread(
            [&]()
            {
                ros::Rate r(3.3);
                while (ros::ok())
                {
                    { // Align distance measurements
                        lock_guard<mutex> lock1(dist_meas_raw_mtx);
                        lock_guard<mutex> lock2(dist_meas_mtx);
                        for (auto &[jupiter_id, jupiters] : swarm_dist_meas_raw)
                        {
                            while (!jupiters.empty())
                            {
                                const nlink_parser::LinktrackNodeframe3 &jupiter_raw = jupiters.front().distance_meas;
                                CHECK(jupiter_raw.id == jupiter_id);

                                ros::Time jupiter_time = jupiters.front().header.stamp;
                                geometry_msgs::Pose pose_from_odom;
                                bool jupiter_odom_found = alignMeasAndOdomByTimestamp(jupiter_id, jupiter_time, pose_from_odom);
                                if (!jupiter_odom_found)
                                {
                                    jupiters.pop_front();
                                    continue;
                                }

                                Jupiter jupiter;
                                jupiter.id = jupiter_id;
                                jupiter.position = Vector3d(pose_from_odom.position.x,
                                                            pose_from_odom.position.y,
                                                            pose_from_odom.position.z);
                                jupiter.orientation = Quaterniond(pose_from_odom.orientation.w,
                                                                  pose_from_odom.orientation.x,
                                                                  pose_from_odom.orientation.y,
                                                                  pose_from_odom.orientation.z);

                                for (const nlink_parser::LinktrackNode2 &ganymede_raw : jupiter_raw.nodes)
                                {
                                    int ganymede_id = ganymede_raw.id;
                                    CHECK(drone_id.find(ganymede_id) != drone_id.end() ||
                                          uwb_anchor_id.find(ganymede_id) != uwb_anchor_id.end());

                                    Ganymede ganymede;
                                    ganymede.id = ganymede_id;

                                    bool ganymede_is_uwb_anchor = uwb_anchor_id.find(ganymede_id) != uwb_anchor_id.end();
                                    if (ganymede_is_uwb_anchor)
                                    {
                                        ganymede.position = uwb_anchor_position[ganymede_id];
                                        ganymede.orientation = Quaterniond(1.0, 0.0, 0.0, 0.0);
                                    }
                                    else
                                    {
                                        ros::Time ganymede_time = jupiter_time;

                                        geometry_msgs::Pose pose_from_odom;
                                        bool ganymede_odom_found = alignMeasAndOdomByTimestamp(ganymede_id, ganymede_time, pose_from_odom);
                                        if (!ganymede_odom_found)
                                            continue;

                                        ganymede.position = Vector3d(pose_from_odom.position.x,
                                                                     pose_from_odom.position.y,
                                                                     pose_from_odom.position.z);
                                        ganymede.orientation = Quaterniond(pose_from_odom.orientation.w,
                                                                           pose_from_odom.orientation.x,
                                                                           pose_from_odom.orientation.y,
                                                                           pose_from_odom.orientation.z);
                                    }

                                    double distance = ganymede_raw.dis;
                                    DistMeas dist_meas;
                                    dist_meas.timestamp = jupiter_time;
                                    dist_meas.jupiter = jupiter;
                                    dist_meas.ganymede = ganymede;
                                    dist_meas.distance = distance;

                                    swarm_dist_meas.push_back(dist_meas);
                                    // TODO: a lot of copy so far #^# Try to use move instead

                                    fresh_dist_meas_ready = true;
                                }
                                jupiters.pop_front();
                            }
                        }
                        // Keep 6 seconds measurements in the sliding window FIXME: swarm_dist_meas is not ordered in time #^#
                        if (fresh_dist_meas_ready)
                            while (swarm_dist_meas.front().timestamp < swarm_dist_meas.back().timestamp - ros::Duration(6.0))
                                swarm_dist_meas.pop_front();
                    }

                    { // Align anonymous bearing measurements
                        lock_guard<mutex> lock1(anonymous_bearing_meas_raw_mtx);
                        lock_guard<mutex> lock2(anonymous_bearing_meas_mtx);
                        for (auto &[jupiter_id, jupiters] : swarm_anonymous_bearing_meas_raw)
                        {
                            while (!jupiters.empty())
                            {
                                const geometry_msgs::PointStamped &jupiter_raw = jupiters.front().anonymous_bearing;
                                CHECK(jupiters.front().id == jupiter_id);

                                ros::Time jupiter_time = jupiter_raw.header.stamp;
                                geometry_msgs::Pose pose_from_odom;
                                bool jupiter_odom_found = alignMeasAndOdomByTimestamp(jupiter_id, jupiter_time, pose_from_odom);
                                if (!jupiter_odom_found)
                                {
                                    jupiters.pop_front();
                                    continue;
                                }

                                Jupiter jupiter;
                                jupiter.id = jupiter_id;
                                jupiter.position = Vector3d(pose_from_odom.position.x,
                                                            pose_from_odom.position.y,
                                                            pose_from_odom.position.z);
                                jupiter.orientation = Quaterniond(pose_from_odom.orientation.w,
                                                                  pose_from_odom.orientation.x,
                                                                  pose_from_odom.orientation.y,
                                                                  pose_from_odom.orientation.z);

                                Vector3d bearing(jupiter_raw.point.x,
                                                 jupiter_raw.point.y,
                                                 jupiter_raw.point.z);
                                CHECK_DOUBLE_EQ(bearing.norm(), 1.0);

                                AnonymousBearingMeas bearing_meas;
                                bearing_meas.timestamp = jupiter_time;
                                bearing_meas.jupiter = jupiter;
                                bearing_meas.bearing = bearing;

                                swarm_anonymous_bearing_meas.push_back(bearing_meas);
                                // TODO: a lot of copy so far #^# Use move instead

                                fresh_anonymous_bearing_meas_ready = true;
                                jupiters.pop_front();
                            }
                        }
                    }

                    r.sleep();
                }
            });

        map<int, Drift> swarm_drift; // Decision variable
        for (int id : drone_id)
            swarm_drift[id] = Drift(); // Initialize to zero
        auto applyDrift = [](const Drift &drift, Vector3d &odom_p, Quaterniond &odom_q)
        {
            Quaterniond drift_q(AngleAxisd(drift.yaw, Vector3d::UnitZ()));
            odom_p = drift_q * odom_p + drift.translation;
            odom_q = drift_q * odom_q;
        };

        double regularization_cost_weight = (double)param["regularization_cost_weight"];
        double distance_cost_weight = (double)param["distance_cost_weight"];
        double bearing_cost_weight = (double)param["bearing_cost_weight"];
        bool optimize_z = (int)param["optimize_z"];
        double max_deanonymization_bearing_angle = (double)param["max_deanonymization_bearing_angle"];

        auto optimizeOverDistanceMeas = [&]()
        {
            ceres::Problem problem;

            // Regularization cost: keep the drift close to initial value
            for (auto &[id, drift] : swarm_drift)
            {
                ceres::CostFunction *regularization_cost = RegularizationCostFunctor::Create(drift.translation,
                                                                                             drift.yaw,
                                                                                             regularization_cost_weight,
                                                                                             optimize_z);

                problem.AddResidualBlock(regularization_cost, nullptr,
                                         drift.translation.data(), &drift.yaw);
            }

            { // Distance measurement cost
                lock_guard<mutex> lock(dist_meas_mtx);
                for (const DistMeas &dist_meas : swarm_dist_meas)
                {
                    // TODO: add outlier detection here

                    // Keep 6 seconds measurements in the sliding window
                    if (dist_meas.timestamp < swarm_dist_meas.back().timestamp - ros::Duration(6.0))
                        continue;

                    bool ganymede_is_uwb_anchor = uwb_anchor_id.find(dist_meas.ganymede.id) != uwb_anchor_id.end();
                    if (ganymede_is_uwb_anchor)
                    {
                        ceres::CostFunction *distance_cost =
                            DistanceCostFunctor::CreateAnchorCost(dist_meas.jupiter.position,
                                                                  dist_meas.ganymede.position,
                                                                  dist_meas.distance,
                                                                  distance_cost_weight,
                                                                  optimize_z);

                        ceres::LossFunction *huber_kernel = new ceres::HuberLoss(1.0);

                        Drift &jupiter_drift = swarm_drift[dist_meas.jupiter.id];

                        problem.AddResidualBlock(distance_cost, huber_kernel,
                                                 jupiter_drift.translation.data(), &jupiter_drift.yaw);
                    }
                    else
                    {
                        ceres::CostFunction *distance_cost = DistanceCostFunctor::Create(dist_meas.jupiter.position,
                                                                                         dist_meas.ganymede.position,
                                                                                         dist_meas.distance,
                                                                                         distance_cost_weight,
                                                                                         optimize_z);

                        ceres::LossFunction *huber_kernel = new ceres::HuberLoss(1.0);

                        Drift &jupiter_drift = swarm_drift[dist_meas.jupiter.id];
                        Drift &ganymede_drift = swarm_drift[dist_meas.ganymede.id];

                        problem.AddResidualBlock(distance_cost, huber_kernel,
                                                 jupiter_drift.translation.data(), &jupiter_drift.yaw,
                                                 ganymede_drift.translation.data(), &ganymede_drift.yaw);
                    }
                }
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            cout << summary.FullReport() << endl;
        };

        deque<BearingMeas> swarm_bearing_meas;
        atomic<bool> fresh_bearing_meas_ready{false};
        auto deanonymizeBearingMeas = [&]()
        {
            {
                lock_guard<mutex> lock1(anonymous_bearing_meas_mtx);
                lock_guard<mutex> lock2(bearing_meas_mtx);
                while (!swarm_anonymous_bearing_meas.empty())
                {
                    const AnonymousBearingMeas &anonymous_bearing_meas = swarm_anonymous_bearing_meas.front();

                    int jupiter_id = anonymous_bearing_meas.jupiter.id;
                    CHECK(drone_id.find(jupiter_id) != drone_id.end());

                    ros::Time jupiter_time = anonymous_bearing_meas.timestamp;

                    Vector3d jupiter_p = anonymous_bearing_meas.jupiter.position;
                    Quaterniond jupiter_q = anonymous_bearing_meas.jupiter.orientation;
                    applyDrift(swarm_drift[jupiter_id], jupiter_p, jupiter_q);

                    Vector3d bearing = anonymous_bearing_meas.bearing;
                    CHECK_DOUBLE_EQ(bearing.norm(), 1.0);

                    Ganymede ganymede;
                    double closest_diff = numeric_limits<double>::infinity();

                    for (int ganymede_id : drone_id)
                    {
                        if (ganymede_id == jupiter_id)
                            continue;

                        ros::Time ganymede_time = jupiter_time;
                        geometry_msgs::Pose pose_from_odom;
                        bool ganymede_odom_found = alignMeasAndOdomByTimestamp(ganymede_id, ganymede_time, pose_from_odom);
                        if (!ganymede_odom_found)
                            continue;

                        Vector3d ganymede_p(pose_from_odom.position.x,
                                            pose_from_odom.position.y,
                                            pose_from_odom.position.z);
                        Quaterniond ganymede_q(pose_from_odom.orientation.w,
                                               pose_from_odom.orientation.x,
                                               pose_from_odom.orientation.y,
                                               pose_from_odom.orientation.z);
                        applyDrift(swarm_drift[ganymede_id], ganymede_p, ganymede_q);

                        Vector3d bearing_from_odom = (jupiter_q.inverse() * (ganymede_p - jupiter_p)).normalized();

                        double diff = acos(clamp(bearing.dot(bearing_from_odom), -1.0, 1.0));
                        if (diff < closest_diff)
                        {
                            closest_diff = diff;
                            ganymede.id = ganymede_id;
                            ganymede.position = ganymede_p;
                            ganymede.orientation = ganymede_q;
                        }
                    }

                    if (closest_diff < max_deanonymization_bearing_angle)
                    {
                        BearingMeas bearing_meas;
                        bearing_meas.timestamp = jupiter_time;
                        bearing_meas.jupiter = anonymous_bearing_meas.jupiter;
                        bearing_meas.ganymede = ganymede;
                        bearing_meas.bearing = bearing;

                        swarm_bearing_meas.push_back(bearing_meas);

                        fresh_bearing_meas_ready = true;
                    }
                    else
                    {
                        LOG(WARNING) << "Deanonymization failed for an anonymous bearing measurement from drone " << jupiter_id << " #^#";
                    }
                    swarm_anonymous_bearing_meas.pop_front();
                }
                // Keep 6 seconds measurements in the sliding window FIXME: swarm_bearing_meas is not ordered in time #^#
                if (fresh_bearing_meas_ready)
                    while (swarm_bearing_meas.front().timestamp < swarm_bearing_meas.back().timestamp - ros::Duration(6.0))
                        swarm_bearing_meas.pop_front();
            }
        };

        auto optimizeOverBearingMeas = [&]()
        {
            ceres::Problem problem;

            // Regularization cost: keep the drift close to initial value
            for (auto &[id, drift] : swarm_drift)
            {
                ceres::CostFunction *regularization_cost = RegularizationCostFunctor::Create(drift.translation,
                                                                                             drift.yaw,
                                                                                             regularization_cost_weight,
                                                                                             optimize_z);

                problem.AddResidualBlock(regularization_cost, nullptr,
                                         drift.translation.data(), &drift.yaw);
            }

            { // Distance measurement cost
                lock_guard<mutex> lock(dist_meas_mtx);
                for (const DistMeas &dist_meas : swarm_dist_meas)
                {
                    // TODO: add outlier detection here

                    // Keep 6 seconds measurements in the sliding window
                    if (dist_meas.timestamp < swarm_dist_meas.back().timestamp - ros::Duration(6.0))
                        continue;

                    bool ganymede_is_uwb_anchor = uwb_anchor_id.find(dist_meas.ganymede.id) != uwb_anchor_id.end();
                    if (ganymede_is_uwb_anchor)
                    {
                        ceres::CostFunction *distance_cost =
                            DistanceCostFunctor::CreateAnchorCost(dist_meas.jupiter.position,
                                                                  dist_meas.ganymede.position,
                                                                  dist_meas.distance,
                                                                  distance_cost_weight,
                                                                  optimize_z);

                        ceres::LossFunction *huber_kernel = new ceres::HuberLoss(1.0);

                        Drift &jupiter_drift = swarm_drift[dist_meas.jupiter.id];

                        problem.AddResidualBlock(distance_cost, huber_kernel,
                                                 jupiter_drift.translation.data(), &jupiter_drift.yaw);
                    }
                    else
                    {
                        ceres::CostFunction *distance_cost = DistanceCostFunctor::Create(dist_meas.jupiter.position,
                                                                                         dist_meas.ganymede.position,
                                                                                         dist_meas.distance,
                                                                                         distance_cost_weight,
                                                                                         optimize_z);

                        ceres::LossFunction *huber_kernel = new ceres::HuberLoss(1.0);

                        Drift &jupiter_drift = swarm_drift[dist_meas.jupiter.id];
                        Drift &ganymede_drift = swarm_drift[dist_meas.ganymede.id];

                        problem.AddResidualBlock(distance_cost, huber_kernel,
                                                 jupiter_drift.translation.data(), &jupiter_drift.yaw,
                                                 ganymede_drift.translation.data(), &ganymede_drift.yaw);
                    }
                }
            }

            { // Bearing measurement cost
                lock_guard<mutex> lock(bearing_meas_mtx);
                for (const BearingMeas &bearing_meas : swarm_bearing_meas)
                {
                    // Keep 6 seconds measurements in the sliding window
                    if (bearing_meas.timestamp < swarm_bearing_meas.back().timestamp - ros::Duration(6.0))
                        continue;

                    ceres::CostFunction *bearing_cost = BearingCostFunctor::Create(bearing_meas.jupiter.position,
                                                                                   bearing_meas.jupiter.orientation,
                                                                                   bearing_meas.ganymede.position,
                                                                                   bearing_meas.ganymede.orientation,
                                                                                   bearing_meas.bearing,
                                                                                   bearing_cost_weight,
                                                                                   optimize_z);

                    ceres::LossFunction *huber_kernel = new ceres::HuberLoss(1.0);

                    Drift &jupiter_drift = swarm_drift[bearing_meas.jupiter.id];
                    Drift &ganymede_drift = swarm_drift[bearing_meas.ganymede.id];

                    problem.AddResidualBlock(bearing_cost, huber_kernel,
                                             jupiter_drift.translation.data(), &jupiter_drift.yaw,
                                             ganymede_drift.translation.data(), &ganymede_drift.yaw);
                }
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            cout << summary.FullReport() << endl;
        };

        ros::Publisher swarm_drift_pub = nh.advertise<relative_loc::Drift>((string)param["drift_to_edges_topic"], 1000);
        auto publishDrift = [&]()
        {
            for (auto &[id, drift] : swarm_drift)
            {
                if (id == self_id)
                    continue;

                relative_loc::Drift drift_with_id;
                drift_with_id.id = id;
                drift_with_id.drift.header.stamp = ros::Time::now();

                drift_with_id.drift.pose.position.x = drift.translation.x();
                drift_with_id.drift.pose.position.y = drift.translation.y();
                drift_with_id.drift.pose.position.z = drift.translation.z();

                Quaterniond q(AngleAxisd(drift.yaw, Vector3d::UnitZ()));
                drift_with_id.drift.pose.orientation.w = q.w();
                drift_with_id.drift.pose.orientation.x = q.x();
                drift_with_id.drift.pose.orientation.y = q.y();
                drift_with_id.drift.pose.orientation.z = q.z();

                swarm_drift_pub.publish(drift_with_id);
            }
        };

        ros::Rate optimizaion_rate(3.3);
        bool swarm_drift_updated = false;
        while (ros::ok())
        {
            if (fresh_dist_meas_ready)
            {
                optimizeOverDistanceMeas();
                fresh_dist_meas_ready = false;
                swarm_drift_updated = true;
            }

            if (fresh_anonymous_bearing_meas_ready)
            {
                deanonymizeBearingMeas();
                fresh_anonymous_bearing_meas_ready = false;

                if (fresh_bearing_meas_ready)
                {
                    optimizeOverBearingMeas();
                    fresh_bearing_meas_ready = false;
                    swarm_drift_updated = true;
                }
            }

            if (swarm_drift_updated)
            {
                // Send latest estimated drift to all drones
                {
                    lock_guard<mutex> lock(drift_mtx);
                    drift_p = swarm_drift[self_id].translation;
                    drift_q = Quaterniond(AngleAxisd(swarm_drift[self_id].yaw, Vector3d::UnitZ()));
                }
                publishDrift();
                swarm_drift_updated = false;
            }

            optimizaion_rate.sleep();
        }
    }
    else
    {
        LOG(INFO) << "\033[32m============= This is an edge node of the relative localization system ^^ =============\033[0m";

        auto recvDriftCallback = [&](const relative_loc::Drift::ConstPtr &msg)
        {
            CHECK_EQ(msg->id, self_id);
            if (abs((msg->drift.header.stamp - ros::Time::now()).toSec()) > 1.0)
            {
                LOG(WARNING) << "Timestamp of drift from center is more than 1.0s later (or earlier) than current time @_@";
            }

            lock_guard<mutex> lock(drift_mtx);
            drift_p = Vector3d(msg->drift.pose.position.x,
                               msg->drift.pose.position.y,
                               msg->drift.pose.position.z);
            drift_q = Quaterniond(msg->drift.pose.orientation.w,
                                  msg->drift.pose.orientation.x,
                                  msg->drift.pose.orientation.y,
                                  msg->drift.pose.orientation.z);
        };
        ros::Subscriber drift_sub =
            nh.subscribe<relative_loc::Drift>((string)param["drift_from_center_topic"], 10,
                                              recvDriftCallback,
                                              ros::VoidConstPtr(),
                                              ros::TransportHints().tcpNoDelay());

        ros::spin();
    }

    return 0;
}