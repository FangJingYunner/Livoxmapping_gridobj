#ifndef OBJSEG_GRID_OBJ_H
#define OBJSEG_GRID_OBJ_H

#include <ros/ros.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Dense>
#include <nav_msgs/GridCells.h>
#include <chrono>
#include <tf/LinearMath/Quaternion.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h> //滤波相关
#include <pcl/filters/random_sample.h> //随机降采样
#include <pcl/console/time.h> //pcl测时间函数
#include<pcl/filters/passthrough.h> //pcl直通滤波
#include <pcl/common/common.h>
#include <math.h>
#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>
// #include <pcl/filters/morphological_filter.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include<pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/random_sample.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/angles.h>
#include <geometry_msgs/TwistStamped.h>
#include <std_msgs/UInt8MultiArray.h>
#include <pcl/common/transforms.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <image_transport/image_transport.h>

using namespace std;
using namespace Eigen;

struct GridCloudNode {
    pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud;
    float cluster_flag;
    float ground_z_elevation;
    float entered;

    GridCloudNode() {
        Cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
        cluster_flag = 0;
        ground_z_elevation = 0;

    }

    void ClearCloud() {
//    Cloud->clear();
        cluster_flag = 0;
        ground_z_elevation = 0;

    }

};

typedef GridCloudNode *GridCloudNodePtr;

class ObjSegGrid_obj {

public:
    ObjSegGrid_obj(ros::NodeHandle &nh_);

    // registered laser scan callback function
    // 输入点云回调函数
    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloud2);

    // 深度图转换为点云
    void Deep2PointCloud(const sensor_msgs::ImageConstPtr &Depth_row_image);

    inline static bool comparez(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2);

    inline static bool comparey(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2);

    inline static bool comparex(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2);

    // 输入图像回调函数，输入深度图
//  void CameraHandler(const sensor_msgs::ImageConstPtr &Depth_row_image);
//  void wheel_callback(const geometry_msgs::TwistStampedConstPtr &twist_msg);
//  void CameraHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloud2);
    // 将点云分布到栅格中
    void ProjectCloudToGrid();

    template<typename PointT>
    inline void extract_initial_seeds_(
            const pcl::PointCloud<PointT> &p_sorted,
            pcl::PointCloud<PointT> &init_seeds);

    template<typename PointT>
    inline void estimate_plane_(const pcl::PointCloud<PointT> &ground);

    // For adaptive
    template<typename PointT>
    // https://blog.csdn.net/qq_38167930/article/details/119165988
    // https://blog.csdn.net/qq_33287871/article/details/106183892
    inline void extract_piecewiseground(
            const pcl::PointCloud<PointT> &src,
            pcl::PointCloud<PointT> &dst,
            pcl::PointCloud<PointT> &non_ground_dst);


    void ComputeGridFeature2();


    void ClusterAndPubObjectGrid();


    void PubFeatureVis();


    void PublishCloud();

    void ClearAll();

    void OdomHandler(const nav_msgs::OdometryConstPtr &Odommsg);


    void UpdateGrid();

    // 运行函数
    void ObjSegGridRun();

    void VisGridState();

    void AddSurroundGrass();

    void PublishGridMap();
    // 变量的定义都在私有变量里
private:
    ros::NodeHandle nh;

    GridCloudNodePtr **GridCloudNodeMap;

    string camera_input_topic;
    string input_Lidar;

    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudCrop;
    pcl::PointCloud<pcl::PointXYZ>::Ptr LidarCloud;
//  pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudTemp_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud;
//  pcl::PointCloud<pcl::PointXYZ>::Ptr total_ground_cloud;
//  pcl::PointCloud<pcl::PointXYZ>::Ptr total_nonground_cloud;
//  pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr ObjPointCloud;
    pcl::PointCloud<pcl::PointXYZ> ground_pc_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr TempCloud;
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColoredCloud2;
//  pcl::PointCloud<pcl::PointXYZI>::Ptr  ObjGrassCloud;
//  pcl::PointCloud<pcl::PointXYZ>::Ptr  EdgeCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr UnderGroundCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr ObjCloudNoGround;
//  pcl::PointCloud<pcl::PointXYZI>::Ptr FinalObjCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr AllGridcloud;

    sensor_msgs::PointCloud2 sensorCloudTemp;

//  pcl::CropBox<pcl::PointXYZ> boundary_box_filter_;
    pcl::CropBox<pcl::PointXYZ> nearby_box_filter_;
//  pcl::CropBox<pcl::PointXYZ> DeepCamera_bbox_filter_;

    ros::Publisher ObjPointCloudPub_;
    ros::Publisher full_cloud_pub_;
    ros::Subscriber subLaserCloud_;
    ros::Publisher XYBoundingMarkArray_pub_;
    ros::Publisher pub_MarkerArrayFeature_;
    ros::Publisher ObjGrassCloudPub_;
    ros::Publisher AllGridCloudPub_;
    ros::Publisher ObjCloudNoGroundPub_;
    ros::Subscriber subOdomAftMapped;

    ros::Publisher GridOccupy;
    ros::Publisher GridFree;
    ros::Publisher GridEmpty;

    image_transport::Publisher GridMapPub;
    image_transport::ImageTransport it;

    visualization_msgs::MarkerArray bounding_marker_array;

    float min_z_elevation = 0.1;
    float LidarHigh = 0.508;

    bool newlaserCloud = false;

    float lidarXAxis = 20;
    float lidarYAxis = 20;

    float MinCarDis = 0.6;

    float GridSize;
    int GridSizeInverse = float(1.0 / GridSize);
    int GridCloudWidth = 201;
    int GridCloudHalfWidth = (GridCloudWidth - 1) / 2;
    int minGridCloudNum = 25;


    int ground_filter_rate = 3;
    int point_filter_rate = 1;
    float ground_height = -0.44;
    float NoDownSampleThr = 9;


    //param for ground plane fit
    int num_lpr_ = 15;
    float th_seeds_ = 0.1;
    int num_iter_ = 3;
    Eigen::Matrix3f cov_;
    Eigen::Vector4f pc_mean_;
    Eigen::VectorXf singular_values_;
    Eigen::MatrixXf normal_;
    float th_dist_d_;
    float d_;
    float th_dist_ = 0.2;
    float uprightness_thr_ = 0.85; //垂直度为30度
    float elevation_thr_ = 0.4;
    float outlier_pc = 0.001;
    float total_score_thr;

    bool UseLidar = false;
    bool HighGrass = false;
    bool Grass = false;
    bool plane = false;
    float SensorHigh;
    std_msgs::Header laser_Stamp;
    tf::Vector3 Translate;
    geometry_msgs::Pose currentpose;
    cv::Mat GridMap;
};

#endif
