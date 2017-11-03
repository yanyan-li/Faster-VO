/*******************************************

*******************************************/

#include "config.h"
#include "visual_odometry.h"

#include <fstream>
#include <boost/timer.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>  
#include <g2o/core/optimization_algorithm_levenberg.h>  
#include <g2o/solvers/csparse/linear_solver_csparse.h>  
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/block_solver.h>


using namespace g2o;
using namespace cv;
using namespace std;

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"please use correct input: bin/faster_vo config/parameter_file"<<endl;
        return 1;
    }

    // inpute parameter of camera and images
    Faster_VO::Config::setParameterFile ( argv[1] );
    string dataset_dir = Faster_VO::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }
    
    Faster_VO::VisualOdometry::Ptr f_vo ( new Faster_VO::VisualOdometry );
    
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    int startIndex=0;
    int endIndex=0;
    while ( !fin.eof() )
    {
        //文件的格式是 时间
		string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;

        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }

    endIndex=rgb_files.size();
    Faster_VO::Camera::Ptr camera ( new Faster_VO::Camera );

    // visualization---Camera pose and landmarks
    cv::viz::Viz3d vis ( "FASTER RGB-D VO" );

    cv::viz::WCoordinateSystem camera_coor ( 0.5 );
    cv::Point3d cam_pos ( 0, -1.0, -1.0 ), cam_focal_point ( 0,0,0 ), cam_y_dir ( 0,1,0 );
    cv::Affine3d cam_pose = cv::viz::makeCameraPose ( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose ( cam_pose );

    camera_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 2.0 );

    vis.showWidget ( "Camera", camera_coor );
	//pose graph
    typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
    typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 
    g2o::SparseOptimizer    optimizer;
    optimizer.setVerbose( false );
    
	ofstream outfile("../output/pose.txt");
    for ( int i=0; i<rgb_files.size(); i++ )
    {

		g2o::VertexSE3Expmap* node = new g2o::VertexSE3Expmap();
		node->setId(i);
		node->setEstimate(g2o::SE3Quat());
		if(i==1)
		{
			node->setFixed(true);
			optimizer.addVertex(node);
		}
		else
		{
			g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
         	edge->vertices() [0] = optimizer.vertex( i-1 );
        	edge->vertices() [1] = optimizer.vertex( i );
			optimizer.addEdge(edge);
		}
		
        node->setFixed(true);
        optimizer.addVertex(node);
		

        cout<<"****** loop "<<i<<" ******"<<endl;
        Mat color = cv::imread ( rgb_files[i] );
        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        Faster_VO::Frame::Ptr pFrame = Faster_VO::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_  = color;
        pFrame->depth_  = depth;
        pFrame->time_stamp_ = rgb_times[i];
        boost::timer timer;



     	f_vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed() <<endl;
		//LOST CASE
        if ( f_vo->state_ ==Faster_VO::VisualOdometry::LOST )
            break;
	    SE3 Twc = pFrame->T_c_w_.inverse();

        cv::Affine3d M (
            cv::Affine3d::Mat3 (
                Twc.rotation_matrix() ( 0,0 ), Twc.rotation_matrix() ( 0,1 ), Twc.rotation_matrix() ( 0,2 ),
                Twc.rotation_matrix() ( 1,0 ), Twc.rotation_matrix() ( 1,1 ), Twc.rotation_matrix() ( 1,2 ),
                Twc.rotation_matrix() ( 2,0 ), Twc.rotation_matrix() ( 2,1 ), Twc.rotation_matrix() ( 2,2 )
            ),
        cv::Affine3d::Vec3 (
                Twc.translation() ( 0,0 ), Twc.translation() ( 1,0 ), Twc.translation() ( 2,0 )
            )
        );
		//输出t
		cout<<"T:"<<pFrame->T_c_w_.translation() ( 0,0 )<<","<< Twc.translation() ( 1,0 )<<","<< Twc.translation() ( 2,0 )<<endl;
        Mat img_show = color.clone();
        Mat cloud(1,1889,CV_32FC3);
        Point3f *data=cloud.ptr<cv::Point3f>();
        int landmark_number=0;
		for ( auto& pt:f_vo->map_->map_points_ )
        {
            Faster_VO::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel ( p->pos_, pFrame->T_c_w_ );
            cv::circle ( img_show, cv::Point2f ( pixel ( 0,0 ),pixel ( 1,0 ) ), 5, cv::Scalar ( 0,255,0 ), 2 );
	        landmark_number++;
			data[landmark_number].x=pixel(0,0);
			data[landmark_number].y=pixel(1,0);
			data[landmark_number].z=0;
           
        }

        cv::viz::WCloud landmark_cloud(cloud,cv::viz::Color::red());
		landmark_cloud.setRenderingProperty ( cv::viz::POINT_SIZE, 5.0 );

        cv::imshow ( "image", img_show );
        cv::waitKey ( 1 );
        vis.setWidgetPose ( "Camera", M );
        vis.showWidget("landmark",landmark_cloud); 
	    vis.spinOnce ( 1, true );
        cout<<endl;
    }
	outfile.close();
    optimizer.save("../output/ba.g2o");
    return 0;
}
