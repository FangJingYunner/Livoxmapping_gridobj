#include "objseg_grid_obj.h"

#define Random(x) (rand() % x)


using namespace std;
using namespace Eigen;
using namespace cv;

ObjSegGrid_obj::ObjSegGrid_obj(ros::NodeHandle &nh_) : nh(nh_), it(nh_) {


    // 读取输入点云topic
    nh.param<string>("/objseg_gridclass/input_Lidar", input_Lidar, "/velodyne_cloud_registered");
    // 读取输入图像topic
//    nh.param<string>("/objseg_gridclass/camera_input_topic", camera_input_topic, "/camera/depth/image_rect_raw");
//    nh.param<string>("/objseg_gridclass/camera_input_topic", camera_input_topic, "/camera/depth/color/points");

    // 读取栅格大小
    nh.param<float>("/objseg_gridclass/GridSize", GridSize, 0.25);
    // 读取总的点云格子一边数量
    nh.param<int>("/objseg_gridclass/GridCloudWidth", GridCloudWidth, 101);
    // 激光雷达安装高度
    nh.param<float>("/objseg_gridclass/LidarHigh", LidarHigh, 0.508);
    // 栅格内最少的点云数量
    nh.param<int>("/objseg_gridclass/minGridCloudNum", minGridCloudNum, 20);
    // 高度阈值
    nh.param<float>("/objseg_gridclass/elevation_thr", elevation_thr_, 0.4);
    // 角度阈值
    nh.param<float>("/objseg_gridclass/uprightness_thr", uprightness_thr_, 0.85);
    // low point representative 低的点数？？？
    nh.param<int>("/objseg_gridclass/num_lpr", num_lpr_, 15);
    // 距离？？？
    nh.param<float>("/objseg_gridclass/th_dist", th_dist_, 0.1);
    // 迭代次数
    nh.param<int>("/objseg_gridclass/num_iter", num_iter_, 5);
    // 种子数
    nh.param<float>("/objseg_gridclass/th_seeds", th_seeds_, 0.1);
    //小于min_z_elevation高度直接判定为地面
    nh.param<float>("/objseg_gridclass/min_z_elevation", min_z_elevation, 0.1);

    // 相机内参标定值

    nh.param<float>("/objseg_gridclass/outlier_pc", outlier_pc, 0.01);


    // 不均匀采样率
    nh.param<int>("/objseg_gridclass/ground_filter_rate", ground_filter_rate, 3);
    nh.param<int>("/objseg_gridclass/point_filter_rate", point_filter_rate, 1);
    // 地面位置，z=-0.35
    nh.param<float>("/objseg_gridclass/ground_height", ground_height, -0.4);

    // 激光雷达使用范围
    nh.param<float>("/objseg_gridclass/lidarXAxis", lidarXAxis, 10);
    nh.param<float>("/objseg_gridclass/lidarYAxis", lidarYAxis, 10);
    cout << "lidarXAxis:" << lidarXAxis << " lidarYAxis:" << lidarYAxis << endl;

    nh.param<float>("/objseg_gridclass/MinCarDis", MinCarDis, 0.6);
    nh.param<float>("/objseg_gridclass/total_score_thr", total_score_thr, 1.5);

    nh.param<bool>("/objseg_gridclass/UseLidar", UseLidar, true);//使用雷达或者深度相机检测
    nh.param<bool>("/objseg_gridclass/HighGrass", HighGrass, false);
    nh.param<bool>("/objseg_gridclass/Grass", Grass, false);
    nh.param<bool>("/objseg_gridclass/plane", plane, false);


    std::cout << "UseLidar" << UseLidar << endl;
    std::cout << "HighGrass" << HighGrass << endl;
    std::cout << "Grass" << Grass << endl;
    std::cout << "plane" << plane << endl;


    // 创立新点云
    laserCloudCrop.reset(new pcl::PointCloud<pcl::PointXYZ>());
    LidarCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    ObjPointCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    AllGridcloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    if (UseLidar) {
        SensorHigh = LidarHigh;
    }


    // 只保留xy方向正负20m，z方向2m以下的点云 只保留地面及以下点
//    boundary_box_filter_.setMin(Eigen::Vector4f(-lidarXAxis, -lidarYAxis, -LidarHigh, 1.0));
//    boundary_box_filter_.setMax(Eigen::Vector4f(lidarXAxis, lidarYAxis, LidarHigh, 1.0));
//    boundary_box_filter_.setMin(Eigen::Vector4f(-10, -10, -10, 1.0));
//    boundary_box_filter_.setMax(Eigen::Vector4f(10, 10, 10, 1.0));

//    boundary_box_filter_.setNegative(false); //保留框内的


    //zvision滤去没有接收到的点，没有接收到的点都是0，0，0，
    // 去除左右细长条
    nearby_box_filter_.setMin(Eigen::Vector4f(-100, -100, -SensorHigh - 0.1, 1.0));
    nearby_box_filter_.setMax(Eigen::Vector4f(100, 100, 1.2, 1.0));
//    nearby_box_filter_.setNegative(true); //保留框外的
    nearby_box_filter_.setNegative(false);

    // 获取点云一半的.一边151个格子,即左边75个格子,右边75个格子
    GridCloudHalfWidth = (GridCloudWidth - 1) / 2;
    // 格子宽度的导数
    GridSizeInverse = 1 / GridSize;

    // 订阅输入点云,运行laserCloudHandler函数，对点云进行初筛，存入全局变量中
    subLaserCloud_ = nh.subscribe<sensor_msgs::PointCloud2>(input_Lidar, 1, &ObjSegGrid_obj::laserCloudHandler, this);
    subOdomAftMapped = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 1, &ObjSegGrid_obj::OdomHandler, this);

    // 发布变量
    ObjPointCloudPub_ = nh.advertise<sensor_msgs::PointCloud2>("/objseg_gridclass/ObjPointCloud", 1);
    full_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/objseg_gridclass/fullPointCloud", 1);
    pub_MarkerArrayFeature_ = nh.advertise<visualization_msgs::MarkerArray>("/objseg_gridclass/MarkerArrayFeature", 1);
    AllGridCloudPub_ = nh.advertise<sensor_msgs::PointCloud2>("/objseg_gridclass/AllGridCloud", 1);
    GridMapPub = it.advertise("/objseg_gridclass/GridMap", 1);

    GridOccupy = nh.advertise<nav_msgs::GridCells>("/GridOccupy", 1);
    GridFree = nh.advertise<nav_msgs::GridCells>("/GridFree", 1);
    GridEmpty = nh.advertise<nav_msgs::GridCells>("/GridEmpty", 1);

    // 创建一个数组,作为栅格地图.GridCloudWidth个位置
    GridCloudNodeMap = new GridCloudNodePtr *[GridCloudWidth];
    // 每一个数组，在创建一个GridCloudWidth个位置的数组
    for (int i = 0; i < GridCloudWidth; i++) {
        GridCloudNodeMap[i] = new GridCloudNodePtr[GridCloudWidth];
        for (int j = 0; j < GridCloudWidth; j++) {
            // 二维数组内每一个点，初始一个栅格对象，里面包含点云、是否聚类等信息，是结构体
            GridCloudNodeMap[i][j] = new GridCloudNode();
        }
    }

    // 初始化完成
    std::cerr << "Init objSeg" << endl;
}


void ObjSegGrid_obj::OdomHandler(const nav_msgs::OdometryConstPtr &Odommsg) {


    currentpose.position.x = Odommsg->pose.pose.position.x;
    currentpose.position.y = Odommsg->pose.pose.position.y;
    currentpose.position.z = Odommsg->pose.pose.position.z;

    cout << "position.x: " << currentpose.position.x << " position.y: " << currentpose.position.y << " position.z: "
         << currentpose.position.z << endl;

}

void ObjSegGrid_obj::laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloud2) {

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    LidarCloud->clear();
    laser_Stamp = laserCloud2->header;

    static int loop = 0;
    static float avg_ms = 0;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    // 从ros中读入点云到变量里
    pcl::fromROSMsg(*laserCloud2, *LidarCloud);
    laser_Stamp.stamp = laserCloud2->header.stamp;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*LidarCloud, *LidarCloud, indices);

//      nearby_box_filter_.setInputCloud(LidarCloud);
//      nearby_box_filter_.filter(*LidarCloud);



    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = end - start;
    avg_ms += fp_ms.count();

    std::cout << "Handlertime: " << fp_ms.count() << "avg Handlertime:" << avg_ms / loop << endl;
    loop++;

    newlaserCloud = true;

}


inline bool ObjSegGrid_obj::comparez(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2) {
    return p1.z < p2.z;
}

inline bool ObjSegGrid_obj::comparey(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2) {
    return p1.y < p2.y;
}

inline bool ObjSegGrid_obj::comparex(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2) {
    return p1.x < p2.x;
}


// 将点云分布到栅格中
void ObjSegGrid_obj::ProjectCloudToGrid() {
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    // laserCloudCrop为随机采样后的点云，这里获取采样后的点云的大小
    int CloudSize = laserCloudCrop->size();
    int indX, indY;

    // 遍历每一个点云
    for (int i = 0; i < CloudSize; i++) {
        // fixme 栅格的中心点好像有偏置 将点云除以单位栅格的长度，得到栅格的id
        indX = floor(GridSizeInverse * (laserCloudCrop->points[i].x + GridSize / 2.0)) + GridCloudHalfWidth;
        indY = floor(GridSizeInverse * (laserCloudCrop->points[i].y + GridSize / 2.0)) + GridCloudHalfWidth;
        // 如果点云不在栅格内，则跳过这个点
        if (indX < 0 || indX > GridCloudWidth - 1 || indY < 0 || indY > GridCloudWidth - 1)
            continue;
        // 向栅格内填充点
        GridCloudNodeMap[indX][indY]->Cloud->push_back(laserCloudCrop->points[i]);
    }



    // 遍历每一个栅格
    for (int indX = 0; indX < GridCloudWidth; indX++) {
        for (int indY = 0; indY < GridCloudWidth; indY++) {

            if (GridCloudNodeMap[indX][indY]->Cloud->points.size() >= minGridCloudNum) {

                // 将点云按z轴数值大小排序z值小的排的位置小，z值大的索引大
                sort(GridCloudNodeMap[indX][indY]->Cloud->points.begin(),
                     GridCloudNodeMap[indX][indY]->Cloud->points.end(),
                     comparez);
                int cloudsize = GridCloudNodeMap[indX][indY]->Cloud->points.size();
                double ground_z_elevation = GridCloudNodeMap[indX][indY]->Cloud->points[cloudsize - 1].z -
                                            GridCloudNodeMap[indX][indY]->Cloud->points[0].z;

                GridCloudNodeMap[indX][indY]->ground_z_elevation = ground_z_elevation;

//                GridCloudNodeMap[indX][indY]->ground_z_elevation = GridCloudNodeMap[indX][indY]->Cloud->points[cloudsize - 1].z + 0.2;

//                cout<<"indx:"<<indX<<"indY:"<<indY<<"elev:"<<ground_z_elevation<<endl;


            }
        }
    }
}


template<typename PointT>
inline void ObjSegGrid_obj::extract_initial_seeds_(
        const pcl::PointCloud<PointT> &p_sorted,
        pcl::PointCloud<PointT> &init_seeds) {

    init_seeds.points.clear();

    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;

    // Calculate the mean height value.
    for (int i = 0; i < p_sorted.points.size() && cnt < num_lpr_; i++) {
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0; // in case divide by 0

    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (int i = 0; i < p_sorted.points.size(); i++) {
        if (p_sorted.points[i].z < lpr_height + th_seeds_) {
            init_seeds.points.push_back(p_sorted.points[i]);
        }
    }
}

template<typename PointT>
inline void ObjSegGrid_obj::estimate_plane_(const pcl::PointCloud<PointT> &ground) {
    pcl::computeMeanAndCovarianceMatrix(ground, cov_, pc_mean_);
    // Singular Value Decomposition: SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);
    singular_values_ = svd.singularValues();

    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean_.head<3>();

    // according to normal.T*[x,y,z] = -d
    d_ = -(normal_.transpose() * seeds_mean)(0, 0);
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;
}

// For adaptive
template<typename PointT>
// https://blog.csdn.net/qq_38167930/article/details/119165988
// https://blog.csdn.net/qq_33287871/article/details/106183892
inline void ObjSegGrid_obj::extract_piecewiseground(
        const pcl::PointCloud<PointT> &src,
        pcl::PointCloud<PointT> &dst,
        pcl::PointCloud<PointT> &non_ground_dst) {
    // 0. Initialization
    if (!ground_pc_.empty())
        ground_pc_.clear();
    if (!dst.empty())
        dst.clear();
    if (!non_ground_dst.empty())
        non_ground_dst.clear();
    // 1. set seeds!

    // 选取种子点,即选取一个栅格里,z值较小的一群点
    extract_initial_seeds_(src, ground_pc_);
    // 2. Extract ground
    for (int i = 0; i < num_iter_; i++) {
        // 估计平面
        estimate_plane_(ground_pc_);
        ground_pc_.clear();

        //pointcloud to matrix
        Eigen::MatrixXf points(src.points.size(), 3);
        int j = 0;
        for (auto &p : src.points) {
            points.row(j++) << p.x, p.y, p.z;
        }
        // ground plane model
        Eigen::VectorXf result = points * normal_;
        // threshold filter
        for (int r = 0; r < result.rows(); r++) {
            if (i < num_iter_ - 1) {
                if (result[r] < th_dist_d_) {
                    ground_pc_.points.push_back(src[r]);
                }
            } else { // Final stage
                if (result[r] < th_dist_d_) {
                    dst.points.push_back(src[r]);
                }
//          else
//          {
//            non_ground_dst.push_back(src[r]);
//          }
            }
        }


        double ground_z_vec = abs(normal_(2, 0));
        float lenresult = result.size() * 0.7;
        float lendst = dst.points.size();
        if (ground_z_vec < uprightness_thr_)
            non_ground_dst += src;

    }
}


void ObjSegGrid_obj::ComputeGridFeature2() {


    // 遍历每个栅格
    for (int indX = 0; indX < GridCloudWidth; indX++) {
        for (int indY = 0; indY < GridCloudWidth; indY++) {
            // 如果栅格内点云大于最低点云要求
            if (GridCloudNodeMap[indX][indY]->Cloud->size() > minGridCloudNum) {
//                cout<<"indX:"<<indX<<"indY:"<<indY<<" cloudsize:"<<GridCloudNodeMap[indX][indY]->Cloud->size()
//                <<" elev:"<<GridCloudNodeMap[indX][indY]->ground_z_elevation<<endl;
                int cloudsize = GridCloudNodeMap[indX][indY]->Cloud->size();
                if (GridCloudNodeMap[indX][indY]->ground_z_elevation > elevation_thr_ ||
                    GridCloudNodeMap[indX][indY]->Cloud->points[cloudsize - 1].z > elevation_thr_) {
                    GridCloudNodeMap[indX][indY]->cluster_flag = 1;
//                    cout<<"indX:"<<indX<<"indY:"<<indY<<endl;
                    continue;
                }

            }
        }
    }
}


//对栅格进行栅格聚类
void ObjSegGrid_obj::ClusterAndPubObjectGrid() {

    ObjPointCloud->clear();
    //栅格BFS聚类
    int label = 2; // 聚类后的flag，会递增，一个类别用一个标签
    pcl::PointXYZI thispoint;

    bool last_cluster = false;
    // 遍历每一个栅格
    for (int indX = 0; indX < GridCloudWidth; indX++) {
        for (int indY = 0; indY < GridCloudWidth; indY++) {
            // 如果flag==1（有障碍物）
            if (GridCloudNodeMap[indX][indY]->cluster_flag == 1) {
                vector<pair<int, int>> neighbour;
                // 存入当前ID到neighbour
                neighbour.push_back(pair<int, int>(indX, indY));
                GridCloudNodeMap[indX][indY]->cluster_flag = label;
//            cout<<"indX:"<<indX<<"indY:"<<indY<<"label:"<<label<<endl;
                // 如果不为空
                while (!neighbour.empty()) {
                    // 提取出这个栅格
                    pair<int, int> thisgrid = neighbour.back();
                    neighbour.pop_back();
                    int indx = thisgrid.first;
                    int indy = thisgrid.second;

                    // 如果这个栅格的flag=2

//            cout<<"indx:"<<indx<<"indy:"<<indy<<"intensity"<<label<<endl;
                    // 获取栅格内点云的数量
                    int CloudSize = GridCloudNodeMap[indx][indy]->Cloud->size();

                    // 遍历每一个点，将反射率I赋予一类的标签，存入障碍物点云
                    for (int i = 0; i < CloudSize; i++) {
                        thispoint.x = GridCloudNodeMap[indx][indy]->Cloud->points[i].x;
                        thispoint.y = GridCloudNodeMap[indx][indy]->Cloud->points[i].y;
                        thispoint.z = GridCloudNodeMap[indx][indy]->Cloud->points[i].z;
                        thispoint.intensity = label;
                        ObjPointCloud->push_back(thispoint);
                    }

                    // 查找前后左右的8个栅格
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            // 如果这个栅格不等于1，即等于0或者2，即没有障碍物或研究遍历过。
                            // 或者就是当前这个栅格
                            // 或者待选的栅格过界
                            // 则跳过这个栅格
                            if ((dx == 0 && dy == 0) ||
                                indx + dx < 0 || indx + dx > GridCloudWidth - 1 || indy + dy < 0 ||
                                indy + dy > GridCloudWidth - 1 ||
                                (GridCloudNodeMap[indx + dx][indy + dy]->cluster_flag != 1 &&
                                 GridCloudNodeMap[indx + dx][indy + dy]->cluster_flag != 0.5))
                                continue;
                            // 否则将这个栅格加入待查找列表
//                cout<<"indx+dx:"<<indx+dx<<"indy+dy:"<<indy+dy<<"clusterflag:"<<GridCloudNodeMap[indx + dx][indy + dy]->cluster_flag<<endl;
                            neighbour.push_back(pair<int, int>(indx + dx, indy + dy));
                            GridCloudNodeMap[indx + dx][indy + dy]->cluster_flag = label;
//                  cout<<"indx+dx:"<<indx+dx<<"indy+dy:"<<indy+dy<<"label:"<<label<<endl;


                        }
                    }
                }
                label++;
            }

        }
    }
    cout << "obj num:" << label << endl;
}


void ObjSegGrid_obj::PubFeatureVis() {
    visualization_msgs::Marker marker_text;
    visualization_msgs::MarkerArray marker_array_text;
    marker_text.header.frame_id = "livox";
    marker_text.header.stamp = ros::Time::now();
    marker_text.ns = "";
    marker_text.color.r = 1;
    marker_text.color.g = 0;
    marker_text.color.b = 0;
    marker_text.color.a = 1;
    marker_text.scale.z = 0.1;
    marker_text.scale.x = 0.1;
    marker_text.scale.y = 0.1;

    marker_text.lifetime = ros::Duration();
//    bbox_marker.frame_locked = true;
    marker_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker_text.action = visualization_msgs::Marker::ADD;
    marker_text.id = 0;
    geometry_msgs::Pose pose_text;
//    int bias = 3*GridSizeInverse;
    for (int indX = 0; indX < GridCloudWidth; indX++) {
        for (int indY = 0; indY < GridCloudWidth; indY++) {

            if (GridCloudNodeMap[indX][indY]->cluster_flag >= 1) {
                marker_text.color.r = 0;
                marker_text.color.g = 0;
                marker_text.color.b = 1;
                pose_text.position.x = (indX - GridCloudHalfWidth) * GridSize;
                pose_text.position.y = (indY - GridCloudHalfWidth) * GridSize;
                marker_text.pose = pose_text;


                marker_text.text += to_string(indX) + ":" + to_string(indY) + "\n" +
                                    to_string((int) (GridCloudNodeMap[indX][indY]->Cloud->size())) + ":" +
                                    to_string(((int) (GridCloudNodeMap[indX][indY]->ground_z_elevation * 100)));

                marker_array_text.markers.emplace_back(marker_text);
                marker_text.text.clear();
                marker_text.id += 1;
            } else if (GridCloudNodeMap[indX][indY]->ground_z_elevation > min_z_elevation) {
                marker_text.color.r = 0.5;
                marker_text.color.g = 0.6;
                marker_text.color.b = 0.6;
                pose_text.position.x = (indX - GridCloudHalfWidth) * GridSize;
                pose_text.position.y = (indY - GridCloudHalfWidth) * GridSize;
                marker_text.pose = pose_text;


                marker_text.text += to_string(indX) + ":" + to_string(indY) + "\n" +
                                    to_string((int) (GridCloudNodeMap[indX][indY]->Cloud->size())) + ":" +
                                    to_string(((int) (GridCloudNodeMap[indX][indY]->ground_z_elevation * 100)));

                marker_array_text.markers.emplace_back(marker_text);
                marker_text.text.clear();
                marker_text.id += 1;
            } else if (GridCloudNodeMap[indX][indY]->Cloud->size() > minGridCloudNum) {
                marker_text.color.r = 1;
                marker_text.color.g = 0;
                marker_text.color.b = 0;
                pose_text.position.x = (indX - GridCloudHalfWidth) * GridSize;
                pose_text.position.y = (indY - GridCloudHalfWidth) * GridSize;
                marker_text.pose = pose_text;

                marker_text.text += to_string(indX) + ":" + to_string(indY) + "\n" +
                                    to_string((int) (GridCloudNodeMap[indX][indY]->Cloud->size())) + ":" +
                                    to_string(((int) (GridCloudNodeMap[indX][indY]->ground_z_elevation * 100)));
                marker_array_text.markers.emplace_back(marker_text);
                marker_text.text.clear();
                marker_text.id += 1;
            } else {
                marker_text.color.r = 0.3;
                marker_text.color.g = 0.4;
                marker_text.color.b = 0.4;
                pose_text.position.x = (indX - GridCloudHalfWidth) * GridSize;
                pose_text.position.y = (indY - GridCloudHalfWidth) * GridSize;
                marker_text.pose = pose_text;


//                marker_text.text += to_string(indX) + ":"+ to_string(indY) + "\n" + to_string((int)(GridCloudNodeMap[indX][indY]->perpendicularity*100)) + ":" +  to_string((int)(GridCloudNodeMap[indX][indY]->min_z_elev*100)) + "\n" + to_string(((int)(GridCloudNodeMap[indX][indY]->Cloud->points.size()))) + ":"+to_string(((int)(GridCloudNodeMap[indX][indY]->ground_z_elevation*100)));

                marker_text.text += to_string(indX) + ":" + to_string(indY) + "\n" +
                                    to_string((int) (GridCloudNodeMap[indX][indY]->Cloud->size())) + ":" +
                                    to_string(((int) (GridCloudNodeMap[indX][indY]->ground_z_elevation * 100)));

                marker_array_text.markers.emplace_back(marker_text);
                marker_text.text.clear();
                marker_text.id += 1;
            }
        }
    }

    pub_MarkerArrayFeature_.publish(marker_array_text);
}


void ObjSegGrid_obj::ClearAll() {


    for (int indX = 0; indX < GridCloudWidth; indX++) {
        for (int indY = 0; indY < GridCloudWidth; indY++) {
            GridCloudNodeMap[indX][indY]->ClearCloud();
        }
    }

    ObjPointCloud->clear();
    AllGridcloud->clear();
    ground_pc_.clear();

}

void ObjSegGrid_obj::UpdateGrid() {

    pcl::RandomSample<pcl::PointXYZ> rs;    //创建滤波器对象


    for (int i = 0; i < GridCloudWidth; i++) {
        for (int j = 0; j < GridCloudWidth; j++) {
            if (GridCloudNodeMap[i][j]->cluster_flag >= 1) {
                if (GridCloudNodeMap[i][j]->Cloud->size() > 200) {
                    rs.setInputCloud(GridCloudNodeMap[i][j]->Cloud);            //设置待滤波点云
                    rs.setSample(100);                    //设置下采样点云的点数
                    rs.filter(*GridCloudNodeMap[i][j]->Cloud);
                }
            } else if (GridCloudNodeMap[i][j]->Cloud->size() > 50) {
                rs.setInputCloud(GridCloudNodeMap[i][j]->Cloud);            //设置待滤波点云
                rs.setSample(20);                    //设置下采样点云的点数
                rs.filter(*GridCloudNodeMap[i][j]->Cloud);

            }

            *AllGridcloud += *GridCloudNodeMap[i][j]->Cloud;


        }
    }
    cout << "AllGridcloud size:" << AllGridcloud->points.size() << endl;

}


void ObjSegGrid_obj::PublishCloud() {

    if (ObjPointCloudPub_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*ObjPointCloud, sensorCloudTemp);
        sensorCloudTemp.header.stamp = ros::Time::now();
        sensorCloudTemp.header.frame_id = "livox";
        ObjPointCloudPub_.publish(sensorCloudTemp);
    }


    if (full_cloud_pub_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*laserCloudCrop, sensorCloudTemp);
        // sensorCloudTemp.header.stamp = ros::Time::now();
        sensorCloudTemp.header.stamp = laser_Stamp.stamp;
        sensorCloudTemp.header.frame_id = "livox";
        full_cloud_pub_.publish(sensorCloudTemp);
    }


    if (AllGridCloudPub_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*AllGridcloud, sensorCloudTemp);
        sensorCloudTemp.header.stamp = ros::Time::now();
        sensorCloudTemp.header.frame_id = "livox";
        AllGridCloudPub_.publish(sensorCloudTemp);
    }


}

void ObjSegGrid_obj::VisGridState() {

    nav_msgs::GridCells Occupycells;
    nav_msgs::GridCells Freecells;
    nav_msgs::GridCells Emptycells;

    Occupycells.header.frame_id = "livox";
    Occupycells.cell_height = GridSize;
    Occupycells.cell_width = GridSize;

    Freecells.header.frame_id = "livox";
    Freecells.cell_height = GridSize;
    Freecells.cell_width = GridSize;

    Emptycells.header.frame_id = "livox";
    Emptycells.cell_height = GridSize;
    Emptycells.cell_width = GridSize;

    geometry_msgs::Point obstacle;

    for (int indX = 0; indX < GridCloudWidth; indX++) {
        for (int indY = 0; indY < GridCloudWidth; indY++) {

            obstacle.x = (indX - GridCloudHalfWidth) * GridSize;
            obstacle.y = (indY - GridCloudHalfWidth) * GridSize;
            obstacle.z = -0.5;

            if (GridCloudNodeMap[indX][indY]->cluster_flag >= 1) {
                Occupycells.cells.push_back(obstacle);
            } else if (GridCloudNodeMap[indX][indY]->cluster_flag == 0 &&
                       GridCloudNodeMap[indX][indY]->Cloud->size() >= 20) {
                Freecells.cells.push_back(obstacle);
            } else {
                Emptycells.cells.push_back(obstacle);
            }
        }
    }
    GridOccupy.publish(Occupycells);
    GridFree.publish(Freecells);
    GridEmpty.publish(Emptycells);


}

void ObjSegGrid_obj::AddSurroundGrass() {
    for (int indX = 0; indX < GridCloudWidth; indX++) {
        for (int indY = 0; indY < GridCloudWidth; indY++) {
            // 如果栅格内点云大于最低点云要求
            if (GridCloudNodeMap[indX][indY]->cluster_flag == 1) {

                vector<pair<int, int>> neighbour;
                // 存入当前ID到neighbour
                neighbour.push_back(pair<int, int>(indX, indY));

                // 如果不为空
                while (!neighbour.empty()) {
                    // 提取出这个栅格
                    pair<int, int> thisgrid = neighbour.back();
                    neighbour.pop_back();
                    int indx = thisgrid.first;
                    int indy = thisgrid.second;

                    // 查找前后左右的8个栅格
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            // 如果这个栅格不等于1，即等于0或者2，即没有障碍物或研究遍历过。
                            // 或者就是当前这个栅格
                            // 或者待选的栅格过界
                            // 则跳过这个栅格
                            if ((dx == 0 && dy == 0) ||
                                indx + dx < 0 || indx + dx > GridCloudWidth - 1 || indy + dy < 0 ||
                                indy + dy > GridCloudWidth - 1)
                                continue;
                            // 否则将这个栅格加入待查找列表
//                            if(GridCloudNodeMap[indx+dx][indy+dy]->cluster_flag == 0 && GridCloudNodeMap[indx+dx][indy+dy]->ground_z_elevation > grasshigh_mid){//这个栅格比草地高
//                                GridCloudNodeMap[indx+dx][indy+dy]->cluster_flag = 1;
//                                neighbour.push_back(pair<int,int>(indx+dx, indy+dy));
//
//                            }else if(GridCloudNodeMap[indx+dx][indy+dy]->cluster_flag == 0 && GridCloudNodeMap[indx+dx][indy+dy]->ground_z_elevation < grasshigh_mid){//这个栅格比草地矮
//                                GridCloudNodeMap[indx+dx][indy+dy]->cluster_flag = 0.5;
//                            }

                            if (GridCloudNodeMap[indx + dx][indy + dy]->cluster_flag == 0) {
                                GridCloudNodeMap[indx + dx][indy + dy]->cluster_flag = 0.5;

                            }

                        }
                    }
                }
            }
        }
    }
}

void ObjSegGrid_obj::PublishGridMap() {
    GridMap = Mat::zeros(GridCloudWidth, GridCloudWidth, CV_32SC1);


    for (unsigned int indY = 0; indY < GridCloudWidth; ++indY) {
        int *row_c = GridMap.ptr<int>(indY);
        for (unsigned int indX = 0; indX < GridCloudWidth; ++indX) {

            if (GridCloudNodeMap[indX][indY]->cluster_flag >= 1) {

                row_c[indX] = 3 * 60;
                cout << "indX:" << indX << " indY:" << indY << " value:" << row_c[indX] << endl;
            } else if (GridCloudNodeMap[indX][indY]->cluster_flag == 0 &&
                       GridCloudNodeMap[indX][indY]->Cloud->size() >= 20) {

                row_c[indX] = 2 * 60;
                cout << "indX:" << indX << " indY:" << indY << " value:" << row_c[indX] << endl;


            } else {

                row_c[indX] = 1 * 60;
                cout << "indX:" << indX << " indY:" << indY << " value:" << row_c[indX] << endl;


            }

        }
    }
    GridMap.convertTo(GridMap, CV_8U);
    sensor_msgs::ImagePtr mask_msg_front = cv_bridge::CvImage(laser_Stamp, "mono8", GridMap).toImageMsg();


    GridMapPub.publish(mask_msg_front);

}

// 运行函数
void ObjSegGrid_obj::ObjSegGridRun() {

    // 计时
    static float avg_ms = 0;
    static float avg_proj = 0;
    static float avg_feature = 0;
    static float avg_cluster = 0;

    // 计数
    static int loop = 1;

    // 如果收到新数据


    // 如果新点云，则标志位false，表示已经使用
    if (newlaserCloud) {
        newlaserCloud = false;
    } else {
        return;
    }
    // 将带点云和深度图放在一个坐标系下laserCloudTemp
    // laserCloudCrop->clear();

    if (UseLidar && LidarCloud->points.size() > 10) {
        laserCloudCrop = LidarCloud;
    } else return;


    // 输出全部点云数量
    std::cout << "all point num: " << laserCloudCrop->size() << " " << endl;
    // 将点云存到栅格中
    ProjectCloudToGrid();


    // 计算点云特征,给每个格子赋予flag，是不是障碍物
    ComputeGridFeature2();

//    AddSurroundGrass();

    // 聚类发布栅格
    ClusterAndPubObjectGrid();

    PubFeatureVis();
//    PubBoundingBox();


//      std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
//      std::chrono::duration<double, std::milli> fp_ms = end - start;
//      avg_ms += fp_ms.count();
//    avg_proj += fp_ms2.count();
//    avg_feature += fp_ms3.count();
//    avg_cluster += fp_ms4.count();
//      std::cout << "Clustertime: " << fp_ms.count() << " avg_proj time:" << avg_proj / loop << " avg_feature time:" << avg_feature / loop << " avg_cluster time:" << avg_cluster / loop << " avg time:" << avg_ms / loop <<endl;
//      loop++;

    UpdateGrid();
    PublishCloud();
    VisGridState();
    PublishGridMap();

    ClearAll();


}

