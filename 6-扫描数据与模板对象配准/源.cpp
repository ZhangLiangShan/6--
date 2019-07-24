//编译方法
//D:\vs_code\PCL_learn\chap_11_点云配准\6-扫描数据与模板对象配准\x64\Release\6-扫描数据与模板对象配准.exe D:\vs_code\PCL_learn\chap_11_点云配准\6-扫描数据与模板对象配准\x64\Release\data\object_templates.txt D:\vs_code\PCL_learn\chap_11_点云配准\6-扫描数据与模板对象配准\x64\Release\data\person.pcd

#define _CRT_SECURE_NO_WARNINGS

#include <limits>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>

class FeatureCloud
{
public:
	// A bit of shorthand
	typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
	typedef pcl::PointCloud<pcl::Normal> SurfaceNormals;
	typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
	typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

	FeatureCloud() ://构造函数
		search_method_xyz_(new SearchMethod),
		normal_radius_(0.02f),
		feature_radius_(0.02f)
	{}

	~FeatureCloud() {}

	//从输入点云指针传入对象点云数据并处理输入
	void
		setInputCloud(PointCloud::Ptr xyz)
	{
		xyz_ = xyz;
		processInput();
	}

	//加载给定pcd文件名的点云数据到对象并处理输入
	void
		loadInputCloud(const std::string &pcd_file)
	{
		xyz_ = PointCloud::Ptr(new PointCloud);
		pcl::io::loadPCDFile(pcd_file, *xyz_);
		processInput();
	}

	//获取指向点云的指针
	PointCloud::Ptr
		getPointCloud() const
	{
		return (xyz_);
	}

	//获取指向点云法线的指针
	SurfaceNormals::Ptr
		getSurfaceNormals() const
	{
		return (normals_);
	}

	//获取指向点云特征描述子的指针
	LocalFeatures::Ptr
		getLocalFeatures() const
	{
		return (features_);
	}

protected:
	//计算表面法向量和特征描述子
	void
		processInput()
	{
		computeSurfaceNormals();
		computeLocalFeatures();
	}

	//计算表面法向量
	void
		computeSurfaceNormals()
	{
		normals_ = SurfaceNormals::Ptr(new SurfaceNormals);

		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
		norm_est.setInputCloud(xyz_);
		norm_est.setSearchMethod(search_method_xyz_);
		norm_est.setRadiusSearch(normal_radius_);
		norm_est.compute(*normals_);
	}

	//计算FPFH特征描述子
	void
		computeLocalFeatures()
	{
		features_ = LocalFeatures::Ptr(new LocalFeatures);

		pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
		fpfh_est.setInputCloud(xyz_);
		fpfh_est.setInputNormals(normals_);
		fpfh_est.setSearchMethod(search_method_xyz_);
		fpfh_est.setRadiusSearch(feature_radius_);
		fpfh_est.compute(*features_);
	}

private:
	// Point cloud data
	PointCloud::Ptr xyz_;
	SurfaceNormals::Ptr normals_;
	LocalFeatures::Ptr features_;
	SearchMethod::Ptr search_method_xyz_;

	// Parameters
	float normal_radius_;
	float feature_radius_;
};

class TemplateAlignment
{
public:

	// A struct for storing alignment results
	struct Result
	{
		float fitness_score;//配准效果值
		Eigen::Matrix4f final_transformation;//变换矩阵
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW//重载宏
	};

	TemplateAlignment() :
		min_sample_distance_(0.05f),
		max_correspondence_distance_(0.01f*0.01f),
		nr_iterations_(500)
	{
		// Intialize the parameters in the Sample Consensus Intial Alignment (SAC-IA) algorithm
		sac_ia_.setMinSampleDistance(min_sample_distance_);
		sac_ia_.setMaxCorrespondenceDistance(max_correspondence_distance_);
		sac_ia_.setMaximumIterations(nr_iterations_);
	}

	~TemplateAlignment() {}

	//设置模板点云需要对齐的目标点云数据
	void
		setTargetCloud(FeatureCloud &target_cloud)
	{
		target_ = target_cloud;
		sac_ia_.setInputTarget(target_cloud.getPointCloud());
		sac_ia_.setTargetFeatures(target_cloud.getLocalFeatures());
	}

	//添加给定的点云到需要配准的模板序列中
	void
		addTemplateCloud(FeatureCloud &template_cloud)
	{
		templates_.push_back(template_cloud);
	}

	//配准模板点云与setTargetCloud ()设定的目标点云
	void
		align(FeatureCloud &template_cloud, TemplateAlignment::Result &result)
	{
		sac_ia_.setInputSource(template_cloud.getPointCloud());
		sac_ia_.setSourceFeatures(template_cloud.getLocalFeatures());

		pcl::PointCloud<pcl::PointXYZ> registration_output;
		sac_ia_.align(registration_output);

		result.fitness_score = (float)sac_ia_.getFitnessScore(max_correspondence_distance_);
		result.final_transformation = sac_ia_.getFinalTransformation();
	}

	//配准所有的模板点云与setTargetCloud ()设定的目标点云
	void
		alignAll(std::vector<TemplateAlignment::Result, Eigen::aligned_allocator<Result> > &results)
	{
		results.resize(templates_.size());
		for (size_t i = 0; i < templates_.size(); ++i)
		{
			align(templates_[i], results[i]);
		}
	}

	//配准所有模板和目标点云获取最拟合的模板
	int
		findBestAlignment(TemplateAlignment::Result &result)
	{
		// Align all of the templates to the target cloud
		std::vector<Result, Eigen::aligned_allocator<Result> > results;
		alignAll(results);

		// Find the template with the best (lowest) fitness score
		float lowest_score = std::numeric_limits<float>::infinity();
		int best_template = 0;
		for (size_t i = 0; i < results.size(); ++i)
		{
			const Result &r = results[i];
			if (r.fitness_score < lowest_score)
			{
				lowest_score = r.fitness_score;
				best_template = (int)i;
			}
		}

		// Output the best alignment
		result = results[best_template];
		return (best_template);
	}

private:
	// A list of template clouds and the target to which they will be aligned
	std::vector<FeatureCloud> templates_;
	FeatureCloud target_;

	// The Sample Consensus Initial Alignment (SAC-IA) registration routine and its parameters
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
	float min_sample_distance_;
	float max_correspondence_distance_;
	int nr_iterations_;
};

// Align a collection of object templates to a sample point cloud
int
main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("No target PCD file given!\n");
		return (-1);
	}

	// Load the object templates specified in the object_templates.txt file
	std::vector<FeatureCloud> object_templates;
	std::ifstream input_stream(argv[1]);
	object_templates.resize(0);
	std::string pcd_filename;
	while (input_stream.good())
	{
		std::getline(input_stream, pcd_filename);
		if (pcd_filename.empty() || pcd_filename.at(0) == '#') // Skip blank lines or comments
			continue;

		FeatureCloud template_cloud;
		template_cloud.loadInputCloud(pcd_filename);
		object_templates.push_back(template_cloud);
	}
	input_stream.close();

	// Load the target cloud PCD file加载目标点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile(argv[2], *cloud);

	// Preprocess the cloud by...预处理
	// ...removing distant points 直通滤波
	const float depth_limit = 1.0;
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0, depth_limit);
	pass.filter(*cloud);

	// ... and downsampling the point cloud 体素栅格下采样
	const float voxel_grid_size = 0.005f;
	pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
	vox_grid.setInputCloud(cloud);
	vox_grid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
	vox_grid.filter(*cloud);

	// Assign to the target FeatureCloud
	FeatureCloud target_cloud;
	target_cloud.setInputCloud(cloud);

	// Set the TemplateAlignment inputs
	TemplateAlignment template_align;
	for (size_t i = 0; i < object_templates.size(); ++i)
	{
		template_align.addTemplateCloud(object_templates[i]);
	}
	template_align.setTargetCloud(target_cloud);

	// Find the best template alignment
	TemplateAlignment::Result best_alignment;
	int best_index = template_align.findBestAlignment(best_alignment);
	const FeatureCloud &best_template = object_templates[best_index];

	// Print the alignment fitness score (values less than 0.00002 are good)
	printf("Best fitness score: %f\n", best_alignment.fitness_score);

	// Print the rotation matrix and translation vector
	Eigen::Matrix3f rotation = best_alignment.final_transformation.block<3, 3>(0, 0);
	Eigen::Vector3f translation = best_alignment.final_transformation.block<3, 1>(0, 3);

	printf("\n");
	printf("    | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
	printf("\n");
	printf("t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));

	// Save the aligned template for visualization
	pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
	pcl::transformPointCloud(*best_template.getPointCloud(), transformed_cloud, best_alignment.final_transformation);
	pcl::io::savePCDFileBinary("output.pcd", transformed_cloud);

	return (0);
}