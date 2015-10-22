#include "header.h"
#include "runlength.h"

using namespace std;
using namespace cv;
using namespace runlength;
int main(){

	Mat input = imread("3-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	
	RunLength coding(input);

	float* feature = new float[1512];
	coding.Dorunlength(feature);

	for (int i = 0; i < 20; i++)
		cout << feature[i] << endl;

	return 0;
}