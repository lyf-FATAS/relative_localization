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
#include "std_msgs/Float32.h"
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
    RegularizationCostFunctor(const Vector3d &initial_translation, double initial_yaw, double prior_weight, double regularization_weight, bool optimize_z, bool optimize_yaw)
        : initial_translation_(initial_translation), initial_yaw_(initial_yaw), prior_weight_(prior_weight), regularization_weight_(regularization_weight), optimize_z_(optimize_z), optimize_yaw_(optimize_yaw) {}

    template <typename T>
    bool operator()(const T *const translation, const T *const yaw, T *residuals) const
    {
        residuals[0] = T(regularization_weight_) * (translation[0] - T(initial_translation_(0))) + T(prior_weight_) * (translation[0] - T(0.0));
        residuals[1] = T(regularization_weight_) * (translation[1] - T(initial_translation_(1))) + T(prior_weight_) * (translation[1] - T(0.0));
        if (optimize_z_)
            residuals[2] = T(regularization_weight_) * (translation[2] - T(initial_translation_(2))) + T(prior_weight_) * (translation[2] - T(0.0));
        else
            residuals[2] = T(0.0);
        if (optimize_yaw_)
            residuals[3] = T(regularization_weight_) * (*yaw - T(initial_yaw_)) + T(prior_weight_) * (*yaw - T(0.0));
        else
            residuals[3] = T(0.0);
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d &initial_translation, double initial_yaw, double prior_weight, double regularization_weight, bool optimize_z, bool optimize_yaw)
    {
        return (new ceres::AutoDiffCostFunction<RegularizationCostFunctor, 4, 3, 1>(
            new RegularizationCostFunctor(initial_translation, initial_yaw, prior_weight, regularization_weight, optimize_z, optimize_yaw)));
    }

private:
    const Vector3d initial_translation_;
    const double initial_yaw_;
    const double prior_weight_;
    const double regularization_weight_;
    const bool optimize_z_;
    const bool optimize_yaw_;
};
struct DistanceCostFunctor
{
    DistanceCostFunctor(const Vector3d &jupiter_odom_p, const Vector3d &ganymede_odom_p,
                        double distance, double weight, bool optimize_z, bool optimize_yaw)
        : jupiter_odom_p_(jupiter_odom_p), ganymede_odom_p_(ganymede_odom_p), distance_(distance), weight_(weight), optimize_z_(optimize_z), optimize_yaw_(optimize_yaw) {}

    template <typename T>
    bool operator()(const T *const jupiter_drift_p, const T *const jupiter_drift_yaw,
                    const T *const ganymede_drift_p, const T *const ganymede_drift_yaw, T *residuals) const
    {
        // Compute drift quaternion and translation for Jupiter
        Map<const Matrix<T, 3, 1>> jupiter_drift_p_(jupiter_drift_p);
        Quaternion<T> jupiter_drift_q = Quaternion<T>(AngleAxis<T>(*jupiter_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Jupiter's odometry position
        Matrix<T, 3, 1> jupiter_p;
        if (optimize_yaw_)
            jupiter_p = jupiter_drift_q * jupiter_odom_p_.cast<T>() + jupiter_drift_p_;
        else
            jupiter_p = jupiter_odom_p_.cast<T>() + jupiter_drift_p_;

        // Compute drift quaternion and translation for Ganymede
        Map<const Matrix<T, 3, 1>> ganymede_drift_p_(ganymede_drift_p);
        Quaternion<T> ganymede_drift_q = Quaternion<T>(AngleAxis<T>(*ganymede_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Ganymede's odometry position
        Matrix<T, 3, 1> ganymede_p;
        if (optimize_yaw_)
            ganymede_p = ganymede_drift_q * ganymede_odom_p_.cast<T>() + ganymede_drift_p_;
        else
            ganymede_p = ganymede_odom_p_.cast<T>() + ganymede_drift_p_;

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
        Matrix<T, 3, 1> jupiter_p;
        if (optimize_yaw_)
            jupiter_p = jupiter_drift_q * jupiter_odom_p_.cast<T>() + jupiter_drift_p_;
        else
            jupiter_p = jupiter_odom_p_.cast<T>() + jupiter_drift_p_;

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
                                       double distance, double weight, bool optimize_z, bool optimize_yaw)
    {
        return (new ceres::AutoDiffCostFunction<DistanceCostFunctor, 1, 3, 1, 3, 1>(
            new DistanceCostFunctor(jupiter_odom_p, ganymede_odom_p, distance, weight, optimize_z, optimize_yaw)));
    }

    static ceres::CostFunction *CreateAnchorCost(const Vector3d &jupiter_odom_p, const Vector3d &ganymede_odom_p,
                                                 double distance, double weight, bool optimize_z, bool optimize_yaw)
    {
        return (new ceres::AutoDiffCostFunction<DistanceCostFunctor, 1, 3, 1>(
            new DistanceCostFunctor(jupiter_odom_p, ganymede_odom_p, distance, weight, optimize_z, optimize_yaw)));
    }

private:
    const Vector3d jupiter_odom_p_;
    const Vector3d ganymede_odom_p_;
    const double distance_;
    const double weight_;
    const bool optimize_z_;
    const bool optimize_yaw_;
};
struct BearingCostFunctor
{
    BearingCostFunctor(const Vector3d &jupiter_odom_p, const Quaterniond &jupiter_odom_q,
                       const Vector3d &ganymede_odom_p, const Quaterniond &ganymede_odom_q,
                       const Vector3d &bearing, double weight, bool optimize_z, bool optimize_yaw)
        : jupiter_odom_p_(jupiter_odom_p), jupiter_odom_q_(jupiter_odom_q), ganymede_odom_p_(ganymede_odom_p), ganymede_odom_q_(ganymede_odom_q), bearing_(bearing), weight_(weight), optimize_z_(optimize_z), optimize_yaw_(optimize_yaw) {}

    template <typename T>
    bool operator()(const T *const jupiter_drift_p, const T *const jupiter_drift_yaw,
                    const T *const ganymede_drift_p, const T *const ganymede_drift_yaw, T *residuals) const
    {
        // Compute drift quaternion and translation for Jupiter
        Map<const Matrix<T, 3, 1>> jupiter_drift_p_(jupiter_drift_p);
        Quaternion<T> jupiter_drift_q = Quaternion<T>(AngleAxis<T>(*jupiter_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Jupiter's odometry
        Matrix<T, 3, 1> jupiter_p;
        Quaternion<T> jupiter_q;
        if (optimize_yaw_)
        {
            jupiter_p = jupiter_drift_q * jupiter_odom_p_.cast<T>() + jupiter_drift_p_;
            jupiter_q = jupiter_drift_q * jupiter_odom_q_.cast<T>();
        }
        else
        {
            jupiter_p = jupiter_odom_p_.cast<T>() + jupiter_drift_p_;
            jupiter_q = jupiter_odom_q_.cast<T>();
        }

        // Compute drift quaternion and translation for Ganymede
        Map<const Matrix<T, 3, 1>> ganymede_drift_p_(ganymede_drift_p);
        Quaternion<T> ganymede_drift_q = Quaternion<T>(AngleAxis<T>(*ganymede_drift_yaw, Matrix<T, 3, 1>::UnitZ()));

        // Apply drift to Ganymede's odometry position
        Matrix<T, 3, 1> ganymede_p;
        if (optimize_yaw_)
            ganymede_p = ganymede_drift_q * ganymede_odom_p_.cast<T>() + ganymede_drift_p_;
        else
            ganymede_p = ganymede_odom_p_.cast<T>() + ganymede_drift_p_;

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
                                       const Vector3d &bearing, double weight, bool optimize_z, bool optimize_yaw)
    {
        return (new ceres::AutoDiffCostFunction<BearingCostFunctor, 1, 3, 1, 3, 1>(
            new BearingCostFunctor(jupiter_odom_p, jupiter_odom_q, ganymede_odom_p, ganymede_odom_q, bearing, weight, optimize_z, optimize_yaw)));
    }

private:
    const Vector3d jupiter_odom_p_;
    const Quaterniond jupiter_odom_q_;
    const Vector3d ganymede_odom_p_;
    const Quaterniond ganymede_odom_q_; // FIXME: unused actually
    const Vector3d bearing_;
    const double weight_;
    const bool optimize_z_;
    const bool optimize_yaw_;
};

int main(int argc, char **argv)
{
    // Quaterniond drift_q_(AngleAxisd(-0.15, Vector3d::UnitZ()));
    // cout << -(drift_q_ * Vector3d(-0.66, -1.04, 0.23)).transpose() << endl;

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
    bool drift_initialized = false;
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
        if (drift_initialized)
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
        }
    };
    ros::Subscriber odom_sub =
        nh.subscribe<nav_msgs::Odometry>((string)param["odom_topic"], 1000,
                                         odomCallback,
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());

    CHECK(getenv("REL_LOC_CENTER_ID") != nullptr);
    int center_id = atoi(getenv("REL_LOC_CENTER_ID"));
    if (center_id == self_id)
    {
        cout << "\033[32m============= This is the central node of the relative localization system *~* =============\033[0m" << endl;

        set<int> drone_id;
        map<int, bool> is_lidar;
        map<int, Vector3d> uwb_extrinsic;
        bool consider_uwb_ext = (int)param["consider_uwb_extrinsic"];
        double uwb_bias = param["uwb_bias"];
        cv::FileNode id_node = param["drone_id_and_uwb_ext"];
        for (auto it = id_node.begin(); it != id_node.end(); ++it)
        {
            int id = (*it)["id"];
            CHECK((drone_id.insert(id)).second);

            bool is_lidar_ = (int)(*it)["is_lidar"];
            is_lidar[id] = is_lidar_;

            vector<double> ext;
            (*it)["uwb_ext"] >> ext;
            CHECK(ext.size() == 3) << "We live in a 3 dimensional world, homie ^^";
            uwb_extrinsic[id] = Vector3d(ext[0], ext[1], ext[2]);
        }
        CHECK(drone_id.find(self_id) != drone_id.end());
        int drone_num = drone_id.size();

        map<int, bool> drift_initialized_;
        for (auto id : drone_id)
        {
            if (is_lidar[id])
                drift_initialized_[id] = true;
            else
                drift_initialized_[id] = false;
        }

        set<int> uwb_anchor_id;
        map<int, Vector3d> uwb_anchor_position;
        cv::FileNode uwb_node = param["uwb_anchor_id_and_position"];
        for (auto it = uwb_node.begin(); it != uwb_node.end(); ++it)
        {
            int id = (*it)["id"];
            CHECK((uwb_anchor_id.insert(id)).second);

            vector<double> pos;
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

        double sliding_window_length = param["sliding_window_length"];

        double swarm_odom_freq = param["swarm_odom_freq"];
        map<int, deque<nav_msgs::Odometry>> swarm_odom_raw;
        auto swarmOdomCallback = [&](const nav_msgs::Odometry::ConstPtr &msg)
        {
            CHECK_EQ(msg->child_frame_id.substr(0, 6), "drone_");

            int id = atoi(msg->child_frame_id.substr(6, 10).c_str());
            CHECK(drone_id.find(id) != drone_id.end()) << "Drone " << id << " is not in the swarm #^#";
            if (abs((msg->header.stamp - ros::Time::now()).toSec()) > 1.0)
                LOG(ERROR) << "Timestamp of a odom from drone " << id << " is more than 1.0s later (or earlier) than current time @_@";

            lock_guard<mutex> lock(swarm_odom_mtx);

            swarm_odom_raw[id].push_back(*msg);

            if (swarm_odom_raw[id].size() > sliding_window_length * swarm_odom_freq)
                swarm_odom_raw[id].pop_front();
        };
        ros::Subscriber swarm_odom_sub =
            nh.subscribe<nav_msgs::Odometry>((string)param["swarm_odom_topic"], 1000,
                                             swarmOdomCallback,
                                             ros::VoidConstPtr(),
                                             ros::TransportHints().tcpNoDelay());

        auto alignOdomAndOdomGtByTimestamp = [&swarm_odom_mtx, &swarm_odom_raw, swarm_odom_freq](const int id, const ros::Time &meas_timestamp, geometry_msgs::Pose &pose_from_odom)
        {
            lock_guard<mutex> lock(swarm_odom_mtx);

            const deque<nav_msgs::Odometry> &odoms = swarm_odom_raw[id];
            if (odoms.empty())
            {
                LOG(WARNING) << "No odom received from drone " << id << " #^#";
                return false;
            }

            const double tolerance = 3.0 / swarm_odom_freq; // 3 odom periods
            if (meas_timestamp < odoms.front().header.stamp - ros::Duration(tolerance) ||
                meas_timestamp > odoms.back().header.stamp + ros::Duration(tolerance))
            {
                LOG(ERROR) << ((meas_timestamp > odoms.back().header.stamp + ros::Duration(tolerance)) ? "Drone " + to_string(id) + "'s odom gt time >> lastest odom time (" + to_string((meas_timestamp - odoms.back().header.stamp).toSec()) + "s) #^#" : "Drone " + to_string(id) + "'s odom gt time << earliest odom time (" + to_string((odoms.front().header.stamp - meas_timestamp).toSec()) + "s) #^#");
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
            if (closest_diff > 2.0 / swarm_odom_freq) // 2 odom periods
                LOG(WARNING) << "Distance from odom gt to drone " << id << "'s nearest odom is " << closest_diff << "s";

            pose_from_odom = (*closest_it).pose.pose;
            return true;
        };
        ros::Publisher swarm_drift_gt_pub = nh.advertise<relative_loc::Drift>((string)param["drift_gt_topic"], 1000);
        auto swarmOdomGtCallback = [&](const nav_msgs::Odometry::ConstPtr &msg)
        {
            CHECK_EQ(msg->child_frame_id.substr(0, 6), "drone_");

            int id = atoi(msg->child_frame_id.substr(6, 10).c_str());
            CHECK(drone_id.find(id) != drone_id.end()) << "Drone " << id << " is not in the swarm #^#";
            if (abs((msg->header.stamp - ros::Time::now()).toSec()) > 1.0)
                LOG(ERROR) << "Timestamp of a groundtruth odom from drone " << id << " is more than 1.0s later (or earlier) than current time @_@";

            Vector3d odom_gt_p(msg->pose.pose.position.x,
                               msg->pose.pose.position.y,
                               msg->pose.pose.position.z);
            Quaterniond odom_gt_q(msg->pose.pose.orientation.w,
                                  msg->pose.pose.orientation.x,
                                  msg->pose.pose.orientation.y,
                                  msg->pose.pose.orientation.z);

            geometry_msgs::Pose odom;
            bool odom_found = alignOdomAndOdomGtByTimestamp(id, msg->header.stamp, odom);
            if (odom_found)
            {
                Vector3d odom_p(odom.position.x,
                                odom.position.y,
                                odom.position.z);
                Quaterniond odom_q(odom.orientation.w,
                                   odom.orientation.x,
                                   odom.orientation.y,
                                   odom.orientation.z);

                Quaterniond drift_gt_q = odom_gt_q * odom_q.inverse();
                Vector3d drift_gt_p = odom_gt_p - drift_gt_q * odom_p;

                relative_loc::Drift drift_gt_msg;
                drift_gt_msg.id = id;
                drift_gt_msg.drift.header.stamp = msg->header.stamp;

                drift_gt_msg.drift.pose.position.x = drift_gt_p.x();
                drift_gt_msg.drift.pose.position.y = drift_gt_p.y();
                drift_gt_msg.drift.pose.position.z = drift_gt_p.z();
                drift_gt_msg.drift.pose.orientation.w = drift_gt_q.w();
                drift_gt_msg.drift.pose.orientation.x = drift_gt_q.x();
                drift_gt_msg.drift.pose.orientation.y = drift_gt_q.y();
                drift_gt_msg.drift.pose.orientation.z = drift_gt_q.z();

                swarm_drift_gt_pub.publish(drift_gt_msg);
            }
        };
        ros::Subscriber swarm_odom_gt_sub =
            nh.subscribe<nav_msgs::Odometry>((string)param["swarm_odom_gt_topic"], 1000,
                                             swarmOdomGtCallback,
                                             ros::VoidConstPtr(),
                                             ros::TransportHints().tcpNoDelay());

        double dist_meas_freq = param["distance_measurement_freq"];
        map<int, deque<relative_loc::DistanceMeas>> swarm_dist_meas_raw;
        auto distanceCallback = [&](const relative_loc::DistanceMeas::ConstPtr &msg)
        {
            int id = msg->distance_meas.id;
            CHECK(drone_id.find(id) != drone_id.end()) << "Drone " << id << " is not in the swarm #^#";
            CHECK(uwb_anchor_id.find(id) == uwb_anchor_id.end()) << "Drone " << id << " is a fxxking uwb anchor #^#";
            if (abs((msg->header.stamp - ros::Time::now()).toSec()) > 1.0)
                LOG(ERROR) << "Timestamp of a distance measurement from drone " << id << " is more than 1.0s later (or earlier) than current time @_@";

            lock_guard<mutex> lock(dist_meas_raw_mtx);

            swarm_dist_meas_raw[id].push_back(*msg);

            if (swarm_dist_meas_raw[id].size() > sliding_window_length * dist_meas_freq)
                swarm_dist_meas_raw[id].pop_front();
        };
        ros::Subscriber distance_sub =
            nh.subscribe<relative_loc::DistanceMeas>((string)param["distance_measurement_topic"], 1000,
                                                     distanceCallback,
                                                     ros::VoidConstPtr(),
                                                     ros::TransportHints().tcpNoDelay());

        double bearing_meas_freq = param["bearing_measurement_freq"];
        map<int, deque<relative_loc::AnonymousBearingMeas>> swarm_anonymous_bearing_meas_raw;
        auto bearingCallback = [&](const relative_loc::AnonymousBearingMeas::ConstPtr &msg)
        {
            CHECK(drone_id.find(msg->id) != drone_id.end()) << "Drone " << msg->id << " is not in the swarm #^#";
            if (abs((msg->anonymous_bearing.header.stamp - ros::Time::now()).toSec()) > 1.0)
                LOG(ERROR) << "Timestamp of a bearing measurement from drone " << msg->id << " is more than 1.0s later (or earlier) than current time @_@";

            lock_guard<mutex> lock(anonymous_bearing_meas_raw_mtx);

            swarm_anonymous_bearing_meas_raw[msg->id].push_back(*msg);

            if (swarm_anonymous_bearing_meas_raw[msg->id].size() > sliding_window_length * bearing_meas_freq)
                swarm_anonymous_bearing_meas_raw[msg->id].pop_front();
        };
        ros::Subscriber bearing_sub =
            nh.subscribe<relative_loc::AnonymousBearingMeas>((string)param["bearing_measurement_topic"], 1000,
                                                             bearingCallback,
                                                             ros::VoidConstPtr(),
                                                             ros::TransportHints().tcpNoDelay());

        ros::AsyncSpinner spinner(4);
        spinner.start();

        map<int, Drift> swarm_drift; // Decision variable
        mutex swarm_drift_mtx;
        for (int id : drone_id)
            swarm_drift[id] = Drift(); // Initialize to zero
        auto applyDrift = [](const Drift &drift, Vector3d &odom_p, Quaterniond &odom_q)
        {
            Quaterniond drift_q(AngleAxisd(drift.yaw, Vector3d::UnitZ()));
            odom_p = drift_q * odom_p + drift.translation;
            odom_q = drift_q * odom_q;
        };

        bool publish_debug_topics = (int)param["publish_debug_topics"];
        ros::Publisher swarm_dist_est_pub = nh.advertise<std_msgs::Float32>((string)param["swarm_dist_est_topic"], 1000);
        ros::Publisher swarm_dist_meas_pub = nh.advertise<std_msgs::Float32>((string)param["swarm_dist_meas_topic"], 1000);
        bool custom_debug_output = (int)param["custom_debug_output"];
        deque<DistMeas> swarm_dist_meas;
        deque<AnonymousBearingMeas> swarm_anonymous_bearing_meas;
        atomic<bool> fresh_dist_meas_ready{false};
        atomic<bool> fresh_anonymous_bearing_meas_ready{false};
        map<int, bool> inform_no_odom;
        for (auto id : drone_id)
            inform_no_odom[id] = true;
        auto alignMeasAndOdomByTimestamp = [&](const int id, const ros::Time &meas_timestamp, geometry_msgs::Pose &pose_from_odom)
        {
            lock_guard<mutex> lock(swarm_odom_mtx);

            const deque<nav_msgs::Odometry> &odoms = swarm_odom_raw[id];
            if (odoms.empty())
            {
                if (inform_no_odom[id])
                {
                    LOG(WARNING) << "No odom received from drone " << id << " #^#";
                    inform_no_odom[id] = false;
                }
                return false;
            }
            else
                inform_no_odom[id] = true;

            const double tolerance = 3.0 / swarm_odom_freq; // 3 odom periods
            if (meas_timestamp < odoms.front().header.stamp - ros::Duration(tolerance) ||
                meas_timestamp > odoms.back().header.stamp + ros::Duration(tolerance))
            {
                LOG(ERROR) << ((meas_timestamp > odoms.back().header.stamp + ros::Duration(tolerance)) ? "Drone " + to_string(id) + "'s distance or bearing measurement time >> lastest odom time (" + to_string((meas_timestamp - odoms.back().header.stamp).toSec()) + "s) #^#" : "Drone " + to_string(id) + "'s distance or bearing measurement time << earliest odom time (" + to_string((odoms.front().header.stamp - meas_timestamp).toSec()) + "s) #^#");
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
            if (closest_diff > 2.0 / swarm_odom_freq) // 2 odom periods
                LOG(WARNING) << "Distance from meas to drone " << id << "'s nearest odom is " << closest_diff << "s";

            pose_from_odom = (*closest_it).pose.pose;
            return true;
        };
        thread align_meas_and_odom_by_timestamp_thread(
            [&]()
            {
                ros::Rate r(max({swarm_odom_freq, dist_meas_freq, bearing_meas_freq}));
                while (ros::ok())
                {
                    { // Align distance measurements with odom
                        lock_guard<mutex> lock1(dist_meas_raw_mtx);
                        lock_guard<mutex> lock2(dist_meas_mtx);
                        for (auto &[jupiter_id, jupiters] : swarm_dist_meas_raw)
                        {
                            while (!jupiters.empty())
                            {
                                const nlink_parser::LinktrackNodeframe3 &jupiter_raw = jupiters.front().distance_meas;

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
                                if (consider_uwb_ext)
                                    jupiter.position += jupiter.orientation * uwb_extrinsic[jupiter_id];

                                for (const nlink_parser::LinktrackNode2 &ganymede_raw : jupiter_raw.nodes)
                                {
                                    int ganymede_id = ganymede_raw.id;
                                    bool ganymede_is_uwb_anchor = uwb_anchor_id.find(ganymede_id) != uwb_anchor_id.end();
                                    CHECK(drone_id.find(ganymede_id) != drone_id.end() || ganymede_is_uwb_anchor);

                                    Ganymede ganymede;
                                    ganymede.id = ganymede_id;

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
                                        if (consider_uwb_ext)
                                            ganymede.position += ganymede.orientation * uwb_extrinsic[ganymede_id];
                                    }

                                    double distance = ganymede_raw.dis + uwb_bias;
                                    DistMeas dist_meas;
                                    dist_meas.timestamp = jupiter_time;
                                    dist_meas.jupiter = jupiter;
                                    dist_meas.ganymede = ganymede;
                                    dist_meas.distance = distance;

                                    swarm_dist_meas.push_back(dist_meas);
                                    // TODO: a lot of copy so far #^# Try to use move instead

                                    fresh_dist_meas_ready = true;

                                    if (publish_debug_topics)
                                    {
                                        Vector3d j_p(jupiter.position);
                                        Quaterniond j_q(jupiter.orientation);
                                        Vector3d g_p(ganymede.position);
                                        Quaterniond g_q(ganymede.orientation);

                                        {
                                            lock_guard<mutex> lock(swarm_drift_mtx);
                                            applyDrift(swarm_drift[jupiter_id], j_p, j_q);
                                            applyDrift(swarm_drift[ganymede_id], g_p, g_q);
                                        }

                                        std_msgs::Float32 dist_est_msg;
                                        dist_est_msg.data = (j_p - g_p).norm();
                                        swarm_dist_est_pub.publish(dist_est_msg);

                                        if (abs(dist_est_msg.data - distance) < 2.5)
                                        {
                                            std_msgs::Float32 dist_meas_msg;
                                            dist_meas_msg.data = distance;
                                            swarm_dist_meas_pub.publish(dist_meas_msg);
                                        }

                                        this_thread::sleep_for(chrono::milliseconds(1));
                                    }

                                    if (custom_debug_output)
                                    {
                                        double diff = distance - (jupiter.position - ganymede.position).norm();
                                        if (diff > 0.1)
                                        {
                                            if (diff > 0.2)
                                                cout << "\033[31mdist meas = " << distance // red
                                                     << ",  gt = " << (jupiter.position - ganymede.position).norm()
                                                     << " (valid only when drift_gt = 0),  diff = " << diff << "\033[0m" << endl;
                                            else
                                                cout << "\033[36mdist meas = " << distance // cyan
                                                     << ",  gt = " << (jupiter.position - ganymede.position).norm()
                                                     << " (valid only when drift_gt = 0),  diff = " << diff << "\033[0m" << endl;
                                        }
                                    }
                                }
                                jupiters.pop_front();
                            }
                        }
                        // FIXME: swarm_dist_meas is not ordered in time #^#
                        if (fresh_dist_meas_ready)
                            while (swarm_dist_meas.front().timestamp < ros::Time::now() - ros::Duration(1.3 * sliding_window_length))
                                swarm_dist_meas.pop_front();
                    }

                    { // Align anonymous bearing measurements with odom
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

        int ceres_verbosity_level = param["ceres_verbosity_level"];
        CHECK(ceres_verbosity_level == 0 || ceres_verbosity_level == 1 || ceres_verbosity_level == 2) << "Invalid Ceres verbosity level #^#";
        double prior_cost_weight = param["prior_cost_weight"];
        double regularization_cost_weight = param["regularization_cost_weight"];
        double distance_cost_weight = param["distance_cost_weight"];
        double bearing_cost_weight = param["bearing_cost_weight"];
        double distance_outlier_thr = param["distance_outlier_threshold"];
        bool optimize_z = (int)param["optimize_z"];
        bool optimize_yaw = (int)param["optimize_yaw"];
        double max_deanonymization_bearing_angle = (double)param["max_deanonymization_bearing_angle"] * 3.14159265358979323846 / 180.0;
        double huber_threshold = param["huber_threshold"];

        map<int, bool> inform_not_enough_dist_meas;
        for (auto id : drone_id)
            inform_not_enough_dist_meas[id] = true;

        auto optimizeOverDistanceMeas = [&]()
        {
            ceres::Problem problem;

            { // Distance measurement cost
                lock_guard<mutex> lock1(dist_meas_mtx);
                lock_guard<mutex> lock2(swarm_drift_mtx);

                map<int, bool> has_enough_dist_meas;
                for (auto id : drone_id)
                    has_enough_dist_meas[id] = false;

                // Used for custom debug output
                int cnt = 1;
                double avg_err_before_opt = 0.0;
                double max_err_before_opt = 0.0;
                double total_err_before_opt = 0.0;
                vector<double> err_before_opt;
                int N_outliers = 0;
                vector<bool> is_outlier;

                for (const DistMeas &dist_meas : swarm_dist_meas)
                {
                    // Keep sliding_window_length seconds measurements in the sliding window
                    if (dist_meas.timestamp < ros::Time::now() - ros::Duration(1.1 * sliding_window_length))
                        continue;

                    // Make sure we have enough measurements in the sliding window
                    if (dist_meas.timestamp < swarm_dist_meas.back().timestamp - ros::Duration(sliding_window_length * 0.5))
                    {
                        drift_initialized_[dist_meas.jupiter.id] = true;
                        drift_initialized_[dist_meas.ganymede.id] = true;
                        has_enough_dist_meas[dist_meas.jupiter.id] = true;
                        has_enough_dist_meas[dist_meas.ganymede.id] = true;
                        inform_not_enough_dist_meas[dist_meas.jupiter.id] = true;
                        inform_not_enough_dist_meas[dist_meas.ganymede.id] = true;
                    }

                    // Used for custom debug output
                    Vector3d jupiter_p(dist_meas.jupiter.position);
                    Quaterniond jupiter_q(dist_meas.jupiter.orientation);
                    Vector3d ganymede_p(dist_meas.ganymede.position);
                    Quaterniond ganymede_q(dist_meas.ganymede.orientation);
                    applyDrift(swarm_drift[dist_meas.jupiter.id], jupiter_p, jupiter_q);
                    applyDrift(swarm_drift[dist_meas.ganymede.id], ganymede_p, ganymede_q);
                    double err = distance_cost_weight * abs(dist_meas.distance - (jupiter_p - ganymede_p).norm());
                    avg_err_before_opt += (err - avg_err_before_opt) / cnt;
                    cnt++;
                    if (err > max_err_before_opt)
                        max_err_before_opt = err;
                    total_err_before_opt += err;
                    err_before_opt.push_back(err);

                    // FIXME: Enhance outlier detection
                    if (abs(dist_meas.distance - (jupiter_p - ganymede_p).norm()) > distance_outlier_thr)
                    {
                        N_outliers++;
                        is_outlier.push_back(true);
                        continue;
                    }
                    is_outlier.push_back(false);

                    bool ganymede_is_uwb_anchor = uwb_anchor_id.find(dist_meas.ganymede.id) != uwb_anchor_id.end();
                    if (ganymede_is_uwb_anchor)
                    {
                        ceres::CostFunction *distance_cost =
                            DistanceCostFunctor::CreateAnchorCost(dist_meas.jupiter.position,
                                                                  dist_meas.ganymede.position,
                                                                  dist_meas.distance,
                                                                  distance_cost_weight,
                                                                  optimize_z,
                                                                  optimize_yaw);

                        ceres::LossFunction *huber_kernel = new ceres::HuberLoss(huber_threshold);

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
                                                                                         optimize_z,
                                                                                         optimize_yaw);

                        ceres::LossFunction *huber_kernel = new ceres::HuberLoss(huber_threshold);

                        Drift &jupiter_drift = swarm_drift[dist_meas.jupiter.id];
                        Drift &ganymede_drift = swarm_drift[dist_meas.ganymede.id];

                        problem.AddResidualBlock(distance_cost, huber_kernel,
                                                 jupiter_drift.translation.data(), &jupiter_drift.yaw,
                                                 ganymede_drift.translation.data(), &ganymede_drift.yaw);
                    }
                }

                // Regularization cost: keep the drift close to initial value
                int N_dist_meas = swarm_dist_meas.size();
                for (auto &[id, drift] : swarm_drift)
                {
                    ceres::CostFunction *regularization_cost = RegularizationCostFunctor::Create(drift.translation,
                                                                                                 drift.yaw,
                                                                                                 N_dist_meas * prior_cost_weight,
                                                                                                 N_dist_meas * regularization_cost_weight,
                                                                                                 optimize_z,
                                                                                                 optimize_yaw);

                    problem.AddResidualBlock(regularization_cost, nullptr,
                                             drift.translation.data(), &drift.yaw);
                }

                for (auto &[id, ready_for_opt] : has_enough_dist_meas)
                {
                    if (!ready_for_opt)
                    {
                        Drift &drift = swarm_drift[id];
                        problem.SetParameterBlockConstant(drift.translation.data());
                        problem.SetParameterBlockConstant(&drift.yaw);
                        if (inform_not_enough_dist_meas[id])
                        {
                            LOG(WARNING) << "Not enough distance measurements associated with drone " << id << ", fixing its drift @_@";
                            inform_not_enough_dist_meas[id] = false;
                        }
                    }
                }

                for (auto &[id, is_lidar_] : is_lidar)
                {
                    if (is_lidar_)
                    {
                        Drift &lidar_drift = swarm_drift[id];
                        problem.SetParameterBlockConstant(lidar_drift.translation.data());
                        problem.SetParameterBlockConstant(&lidar_drift.yaw);
                    }
                }

                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                if (ceres_verbosity_level == 1)
                    options.minimizer_progress_to_stdout = true;

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                if (ceres_verbosity_level == 2)
                    cout << summary.FullReport() << endl;

                if (custom_debug_output)
                {
                    cnt = 1;
                    double avg_err_after_opt = 0.0;
                    double max_err_after_opt = 0.0;
                    double total_err_after_opt = 0.0;
                    CHECK(err_before_opt.size() <= swarm_dist_meas.size());

                    cout << endl
                         << "\033[30;46m[" << ros::Time::now() << "]\033[0m" << endl;
                    cout << "Outlier ratio = " << N_outliers / (double)swarm_dist_meas.size() << endl;
                    cout << fixed << showpoint;
                    cout << setprecision(4);

                    for (const DistMeas &dist_meas : swarm_dist_meas)
                    {
                        if (dist_meas.timestamp < ros::Time::now() - ros::Duration(1.1 * sliding_window_length))
                            continue;

                        Vector3d jupiter_p(dist_meas.jupiter.position);
                        Quaterniond jupiter_q(dist_meas.jupiter.orientation);
                        Vector3d ganymede_p(dist_meas.ganymede.position);
                        Quaterniond ganymede_q(dist_meas.ganymede.orientation);
                        applyDrift(swarm_drift[dist_meas.jupiter.id], jupiter_p, jupiter_q);
                        applyDrift(swarm_drift[dist_meas.ganymede.id], ganymede_p, ganymede_q);

                        double err = distance_cost_weight * abs(dist_meas.distance - (jupiter_p - ganymede_p).norm());
                        avg_err_after_opt += (err - avg_err_after_opt) / cnt;
                        cnt++;
                        if (err > max_err_after_opt)
                            max_err_after_opt = err;
                        total_err_after_opt += err;

                        if (dist_meas.jupiter.id < dist_meas.ganymede.id)
                            cout << "\033[41m" << dist_meas.jupiter.id << "\033[0m " << dist_meas.jupiter.position.x() << " " << dist_meas.jupiter.position.y() << "   "
                                 << dist_meas.ganymede.id << " " << dist_meas.ganymede.position.x() << " " << dist_meas.ganymede.position.y() << "   "
                                 << jupiter_p.x() << " " << jupiter_p.y() << " " << jupiter_p.z() << "   "
                                 << ganymede_p.x() << " " << ganymede_p.y() << " " << ganymede_p.z() << "   ";
                        else
                            cout << dist_meas.ganymede.id << " " << dist_meas.ganymede.position.x() << " " << dist_meas.ganymede.position.y() << "   "
                                 << "\033[41m" << dist_meas.jupiter.id << "\033[0m " << dist_meas.jupiter.position.x() << " " << dist_meas.jupiter.position.y() << "   "
                                 << ganymede_p.x() << " " << ganymede_p.y() << " " << ganymede_p.z() << "   "
                                 << jupiter_p.x() << " " << jupiter_p.y() << " " << jupiter_p.z() << "   ";

                        cout << "\033[36m" << dist_meas.distance << "   ";
                        if (err_before_opt[cnt - 2] < err)
                            cout << "\033[32m" << err_before_opt[cnt - 2] << "\033[0m -> \033[31m" << err << "\033[0m";
                        else
                            cout << "\033[31m" << err_before_opt[cnt - 2] << "\033[0m -> \033[32m" << err << "\033[0m";

                        if (is_outlier[cnt - 2])
                            cout << "  \033[41mis outlier\033[0m" << endl;
                        else
                            cout << endl;
                    }
                    cout << "avg_err = " << avg_err_before_opt
                         << ",  max_err = " << max_err_before_opt
                         << ",  total_err = " << total_err_before_opt << endl;
                    cout << "avg_err = " << avg_err_after_opt
                         << ",  max_err = " << max_err_after_opt
                         << ",  total_err = " << total_err_after_opt << endl
                         << endl;
                }
            }
        };

        deque<BearingMeas> swarm_bearing_meas;
        atomic<bool> fresh_bearing_meas_ready{false};
        auto deanonymizeBearingMeas = [&]()
        {
            {
                lock_guard<mutex> lock1(anonymous_bearing_meas_mtx);
                lock_guard<mutex> lock2(bearing_meas_mtx);
                lock_guard<mutex> lock3(swarm_drift_mtx);
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
                        LOG(ERROR) << "Deanonymization failed (closest_diff = " << closest_diff * 180.0 / 3.14159265358979323846 << " deg) for an anonymous bearing measurement from drone " << jupiter_id << " #^#";
                    }
                    swarm_anonymous_bearing_meas.pop_front();
                }
                // FIXME: swarm_bearing_meas is not ordered in time #^#
                if (fresh_bearing_meas_ready)
                    while (swarm_bearing_meas.front().timestamp < ros::Time::now() - ros::Duration(1.3 * sliding_window_length))
                        swarm_bearing_meas.pop_front();
            }
        };

        auto optimizeOverBearingMeas = [&]()
        {
            ceres::Problem problem;

            lock_guard<mutex> lock(swarm_drift_mtx);

            // Regularization cost: keep the drift close to initial value
            for (auto &[id, drift] : swarm_drift)
            {
                ceres::CostFunction *regularization_cost = RegularizationCostFunctor::Create(drift.translation,
                                                                                             drift.yaw,
                                                                                             prior_cost_weight,
                                                                                             regularization_cost_weight,
                                                                                             optimize_z,
                                                                                             optimize_yaw);

                problem.AddResidualBlock(regularization_cost, nullptr,
                                         drift.translation.data(), &drift.yaw);
            }

            { // Distance measurement cost
                lock_guard<mutex> lock(dist_meas_mtx);
                for (const DistMeas &dist_meas : swarm_dist_meas)
                {
                    // TODO: add outlier detection here

                    // Keep sliding_window_length seconds measurements in the sliding window
                    if (dist_meas.timestamp < ros::Time::now() - ros::Duration(1.1 * sliding_window_length))
                        continue;

                    bool ganymede_is_uwb_anchor = uwb_anchor_id.find(dist_meas.ganymede.id) != uwb_anchor_id.end();
                    if (ganymede_is_uwb_anchor)
                    {
                        ceres::CostFunction *distance_cost =
                            DistanceCostFunctor::CreateAnchorCost(dist_meas.jupiter.position,
                                                                  dist_meas.ganymede.position,
                                                                  dist_meas.distance,
                                                                  distance_cost_weight,
                                                                  optimize_z,
                                                                  optimize_yaw);

                        ceres::LossFunction *huber_kernel = new ceres::HuberLoss(huber_threshold);

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
                                                                                         optimize_z,
                                                                                         optimize_yaw);

                        ceres::LossFunction *huber_kernel = new ceres::HuberLoss(huber_threshold);

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
                    // Keep sliding_window_length seconds measurements in the sliding window
                    if (bearing_meas.timestamp < ros::Time::now() - ros::Duration(1.1 * sliding_window_length))
                        continue;

                    ceres::CostFunction *bearing_cost = BearingCostFunctor::Create(bearing_meas.jupiter.position,
                                                                                   bearing_meas.jupiter.orientation,
                                                                                   bearing_meas.ganymede.position,
                                                                                   bearing_meas.ganymede.orientation,
                                                                                   bearing_meas.bearing,
                                                                                   bearing_cost_weight,
                                                                                   optimize_z,
                                                                                   optimize_yaw);

                    ceres::LossFunction *huber_kernel = new ceres::HuberLoss(huber_threshold);

                    Drift &jupiter_drift = swarm_drift[bearing_meas.jupiter.id];
                    Drift &ganymede_drift = swarm_drift[bearing_meas.ganymede.id];

                    problem.AddResidualBlock(bearing_cost, huber_kernel,
                                             jupiter_drift.translation.data(), &jupiter_drift.yaw,
                                             ganymede_drift.translation.data(), &ganymede_drift.yaw);
                }
            }

            for (auto &[id, is_lidar_] : is_lidar)
            {
                if (is_lidar_)
                {
                    Drift &lidar_drift = swarm_drift[id];
                    problem.SetParameterBlockConstant(lidar_drift.translation.data());
                    problem.SetParameterBlockConstant(&lidar_drift.yaw);
                }
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            if (ceres_verbosity_level == 1)
                options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            if (ceres_verbosity_level == 2)
                cout << summary.FullReport() << endl;
        };

        ros::Publisher swarm_drift_pub = nh.advertise<relative_loc::Drift>((string)param["drift_to_edges_topic"], 1000);
        auto publishDrift = [&]()
        {
            lock_guard<mutex> lock(swarm_drift_mtx);
            for (auto &[id, drift] : swarm_drift)
            {
                if (id == self_id)
                    continue;

                if (!drift_initialized_[id])
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
                this_thread::sleep_for(chrono::milliseconds(1));
            }
        };

        int optimize_per_N_meas = param["optimize_per_N_meas"];
        ros::Rate optimizaion_rate(max({swarm_odom_freq, dist_meas_freq, bearing_meas_freq}) / optimize_per_N_meas);
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
                    lock_guard<mutex> lock1(drift_mtx);
                    lock_guard<mutex> lock2(swarm_drift_mtx);
                    if (drift_initialized_[self_id])
                    {
                        drift_p = swarm_drift[self_id].translation;
                        drift_q = Quaterniond(AngleAxisd(swarm_drift[self_id].yaw, Vector3d::UnitZ()));
                        drift_initialized = true;
                    }
                }
                publishDrift();
                swarm_drift_updated = false;
            }

            optimizaion_rate.sleep();
        }
    }
    else
    {
        cout << "\033[32m============= This is an edge node of the relative localization system ^^ =============\033[0m" << endl;

        auto recvDriftCallback = [&](const relative_loc::Drift::ConstPtr &msg)
        {
            CHECK_EQ(msg->id, self_id);
            if (abs((msg->drift.header.stamp - ros::Time::now()).toSec()) > 1.0)
                LOG(ERROR) << "Timestamp of drift from center is more than 1.0s later (or earlier) than current time @_@";

            lock_guard<mutex> lock(drift_mtx);
            drift_p = Vector3d(msg->drift.pose.position.x,
                               msg->drift.pose.position.y,
                               msg->drift.pose.position.z);
            drift_q = Quaterniond(msg->drift.pose.orientation.w,
                                  msg->drift.pose.orientation.x,
                                  msg->drift.pose.orientation.y,
                                  msg->drift.pose.orientation.z);
            drift_initialized = true;
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