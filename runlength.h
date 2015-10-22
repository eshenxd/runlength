/*
 *  ���빦�ܣ��ԡ�2013 Large-scale document
 *            image retrieval and classification
 *            with runlength �������ᵽ��runlength
 *            ����
 *  �ӿ��趨��input��cv::Mat input_img ��ͨ���Ҷ�ͼ��
 *            output: float feature[1512] �������
 *  �������ߣ�Ethan
 *  ��дʱ�䣺2015-10-21
 *  �޸ļ�¼:
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

		cv::Mat image_in;     //����ͼ��
		cv::Mat image_resize; //��һ��ͼ��
		cv::Mat image_binary; //��ֵͼ��

		cv::Mat width_height;
		cv::Mat Pyramid;
		int rows, cols;
		int nrows, ncols;
	};
}


#endif