/*
 *  代码功能：对《2013 Large-scale document
 *            image retrieval and classification
 *            with runlength 》中所提到的runlength
 *            特征
 *  接口设定：input：cv::Mat input_img 单通道灰度图像
 *            output: float feature[1512] 输出特征
 *  代码作者：Ethan
 *  编写时间：2015-10-21
 *  修改记录:
*/

#ifndef _RUNLENGTH_H_
#define _RUNLENGTH_H_

#include "header.h"

namespace runlength{


	class RunLength
	{
	public:
		RunLength(cv::Mat input_img);
		~RunLength();
		void Dorunlength(float* feature);

	private:
		void mat2vector(cv::Mat in, int* matrix);

		void DoVerticalRunlength(int h, int w, int *im, int *RL);
		void DoHorizontalRunlength(int h, int w, int *im, int *RL);
		void DoDiagonalRunlength(int h, int w, int *im, int *RL);
		void DoAntiDiagonalRunlength(int h, int w, int *im, int *RL);

		void InitMat(cv::Mat& m, float(*p)[4]);

		void img_resize();

		void runlength_encoding(cv::Mat im, float* output);

		cv::Mat image_in;     //输入图像
		cv::Mat image_resize; //归一化图像
		cv::Mat image_binary; //二值图像

		cv::Mat width_height;
		cv::Mat Pyramid;
		int rows, cols;
		int nrows, ncols;
	};
}


#endif