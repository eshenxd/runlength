#include "runlength.h"

using namespace std;
using namespace cv;

namespace runlength{

	float pyramid[21][4] = {
		0, 1, 0, 1,              //1 x 1
		0, 0.5, 0, 0.5,          //2 x 2
		0, 0.5, 0.5, 1,
		0.5, 1, 0, 0.5,
		0.5, 1, 0.5, 1,
		0, 0.25, 0, 0.25,        //4 x 4
		0, 0.25, 0.25, 0.5,
		0, 0.25, 0.5, 0.75,
		0, 0.25, 0.75, 1,
		0.25, 0.5, 0, 0.25,
		0.25, 0.5, 0.25, 0.5,
		0.25, 0.5, 0.5, 0.75,
		0.25, 0.5, 0.75, 1,
		0.5, 0.75, 0, 0.25,
		0.5, 0.75, 0.25, 0.5,
		0.5, 0.75, 0.5, 0.75,
		0.5, 0.75, 0.75, 1,
		0.75, 1, 0, 0.25,
		0.75, 1, 0.25, 0.5,
		0.75, 1, 0.5, 0.75,
		0.75, 1, 0.75, 1
	};

#define nPIXELS 500000.0
	int LUT0[] = { 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 };
#define LENGTH2BIN(P) ((P) >128?8:LUT0[(P)-1])	
	
	/* 构造函数*/
	RunLength::RunLength(Mat image_input){
		rows = image_input.rows;
		cols = image_input.cols;
		image_in = image_input;

		double factor = sqrt(nPIXELS / (cols*rows));
		ncols = cvRound(cols*factor);
		nrows = cvRound(rows*factor);

		width_height = Mat(1, 4, CV_32F);
		float width_height_0[4] = { ncols, ncols, nrows, nrows };

		for (int i = 0; i < width_height.cols; i++)

			width_height.at<float>(0, i) = *(width_height_0 + i);

		Pyramid = Mat(21, 4, CV_32F);
		InitMat(Pyramid, pyramid);

	}
	/* 析构函数*/
	RunLength::~RunLength(){

	}

	/* 图像归一化为固定像素点个数*/
	void RunLength::img_resize(){

		cv::resize(image_in, image_resize, Size(ncols, nrows), 0, 0, CV_INTER_LINEAR);
	}


	
	/* mat to vector*/
	void RunLength::mat2vector(cv::Mat in, int* matrix){
		
		int count = 0;
		for (int i = 0; i < in.rows; i++)
			for (int j = 0; j < in.cols; j++){
			int x = (int)in.at<uchar>(i, j);
			if (x == 0)
				matrix[count++] = 0;
			else
				matrix[count++] = 1;
			}
	}

	void RunLength::DoVerticalRunlength(int h, int w, int *im, int *RL)
	{
		int x, y;
		int *p;
		int currentV, currentL;

		for (x = 0; x < w; ++x)
		{
			p = &im[x*h];
			currentV = *p;
			currentL = 1;
			p++;
			for (y = 1; y < h; ++y, ++p)
			{
				if (*p == currentV)
					currentL++;
				else
				{
					RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
					currentV = *p;
					currentL = 1;
				}
			}
			int tmp = LENGTH2BIN(currentL);
			RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
		}
	}


	void RunLength::DoHorizontalRunlength(int h, int w, int *im, int *RL)
	{
		int x, y;
		int *p;
		int currentV, currentL;

		for (y = 0; y < h; ++y)
		{
			p = &im[y];
			currentV = *p;
			currentL = 1;
			p += h;
			for (x = 1; x < w; ++x, p += h)

			{
				if (*p == currentV)
					currentL++;
				else
				{
					RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
					currentV = *p;
					currentL = 1;
				}
			}
			RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
		}
	}

	void RunLength::DoDiagonalRunlength(int h, int w, int *im, int *RL)
	{
		int x, y;
		int ty, tx;
		int currentV, currentL;

		//Two passes, from up to down and then from left to right

		for (y = 0; y < h; ++y)
		{
			ty = y;
			currentV = im[y];
			currentL = 1;
			for (x = 1; x < w && ty < h; ++x, ++ty)
			{
				if (im[ty + x*h] == currentV)
					currentL++;
				else
				{
					RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
					currentV = im[ty + x*h];
					currentL = 1;
				}
			}
			RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
		}

		for (x = 1; x < w; ++x)
		{
			tx = x;
			currentV = im[x*h];
			currentL = 1;
			for (y = 1; y < h && tx < w; ++y, ++tx)
			{
				if (im[y + tx*h] == currentV)
					currentL++;
				else
				{
					RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
					currentV = im[y + tx*h];
					currentL = 1;
				}
			}
			RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
		}
	}

	void RunLength::DoAntiDiagonalRunlength(int h, int w, int *im, int *RL)
	{
		int x, y;
		int ty, tx;
		int currentV, currentL;

		//Two passes, from up to down and then from left to right

		for (y = 0; y < h; ++y)
		{
			ty = y;
			currentV = im[(w - 1)*h + y];
			currentL = 1;
			for (x = w - 1; x >= 0 && ty < h; --x, ++ty)
			{
				if (im[ty + x*h] == currentV)
					currentL++;
				else
				{
					RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
					currentV = im[ty + x*h];
					currentL = 1;
				}
			}
			RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
		}

		for (x = (w - 2); x >= 0; --x)
		{
			tx = x;
			currentV = im[x*h];
			currentL = 1;
			for (y = 1; y < h && tx >= 0; ++y, --tx)
			{
				if (im[y + tx*h] == currentV)
					currentL++;
				else
				{

					RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
					currentV = im[y + tx*h];
					currentL = 1;
				}
			}
			RL[(9 * currentV) + LENGTH2BIN(currentL)]++;
		}
	}

	/* 提取runlength 特征*/
	void RunLength::Dorunlength(float* feature){
		
		img_resize();
		/* 二值化 */
		adaptiveThreshold(image_resize, image_binary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 31, 15);

		Mat _Pyramid = Mat(1, 4, CV_32F);
		Mat coords = Mat(1, 4, CV_32F);

		for (int i = 0; i < Pyramid.rows; i++){

			_Pyramid = Pyramid(Rect(0, i, 4, 1));
			

			coords = _Pyramid.mul(width_height);
			
			int x1 = (int)coords.at<float>(0, 0);
			int x2 = (int)coords.at<float>(0, 1);
			int y1 = (int)coords.at<float>(0, 2);
			int y2 = (int)coords.at<float>(0, 3);
			Mat test = image_binary(Range(y1, y2), Range(x1, x2));

			runlength_encoding(test, &feature[72 * i]);
		}

	}

	void RunLength::runlength_encoding(cv::Mat im, float* output){
		
		int l = im.rows*im.cols;
		int* a = new int[im.rows*im.cols];
		mat2vector(im, a);

		int* RL = new int[72];
		for (int i = 0; i < 72; i++)
			RL[i] = 0;

		int width = im.cols;
		int heigth = im.rows;

		DoVerticalRunlength(width, heigth, a, RL);
		DoHorizontalRunlength(width, heigth, a, &RL[18]);
		DoDiagonalRunlength(width, heigth, a, &RL[36]);
		DoAntiDiagonalRunlength(width, heigth, a, &RL[54]);

		_Copy_impl(RL, RL + 72, output);

		delete[] a;
		delete[] RL;
	}

	void RunLength::InitMat(Mat& m, float(*p)[4]){
		for (int i = 0; i < m.rows; i++)
			for (int j = 0; j < m.cols; j++)
				m.at<float>(i, j) = *(*(p + i) + j);
	}
	
}