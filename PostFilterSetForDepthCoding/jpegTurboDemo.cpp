#include "util.h"

#undef FAR
#include "jpeglib.h"
//#pragma comment(lib, "jpeg-static.lib")
#pragma comment(lib, "jpeg.lib")
typedef struct {
	struct jpeg_source_mgr pub;	/* public fields */
	JOCTET * buffer;
	unsigned long buffer_length;
} memory_source_mgr;

typedef struct {
	struct jpeg_destination_mgr pub;
	JOCTET * buffer; 
	unsigned long buffer_length;
} memory_destination_mgr;

typedef memory_source_mgr *memory_src_ptr;
typedef memory_destination_mgr *memory_dst_ptr;


METHODDEF(void)
	memory_init_source (j_decompress_ptr cinfo)
{
}

METHODDEF(boolean)
	memory_fill_input_buffer (j_decompress_ptr cinfo)
{
	memory_src_ptr src = (memory_src_ptr) cinfo->src;

	src->buffer[0] = (JOCTET) 0xFF;
	src->buffer[1] = (JOCTET) JPEG_EOI;
	src->pub.next_input_byte = src->buffer;
	src->pub.bytes_in_buffer = 2;
	return TRUE;
}

METHODDEF(void) memory_skip_input_data (j_decompress_ptr cinfo, long num_bytes)
{
	memory_src_ptr src = (memory_src_ptr) cinfo->src;

	if (num_bytes > 0) {
		src->pub.next_input_byte += (size_t) num_bytes;
		src->pub.bytes_in_buffer -= (size_t) num_bytes;
	}
}

METHODDEF(void) memory_term_source (j_decompress_ptr cinfo)
{
}

GLOBAL(void)
	jpeg_memory_src (j_decompress_ptr cinfo, void* data, unsigned long len)
{
	memory_src_ptr src;

	if (cinfo->src == NULL) {	/* first time for this JPEG object? */
		cinfo->src = (struct jpeg_source_mgr *)
			(*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
			sizeof(memory_source_mgr));
		src = (memory_src_ptr) cinfo->src;
		src->buffer = (JOCTET *)
			(*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
			len * sizeof(JOCTET));
	}

	src = (memory_src_ptr) cinfo->src;

	src->pub.init_source = memory_init_source;
	src->pub.fill_input_buffer = memory_fill_input_buffer;
	src->pub.skip_input_data = memory_skip_input_data;
	src->pub.resync_to_restart = jpeg_resync_to_restart; /* use default method */
	src->pub.term_source = memory_term_source;

	src->pub.bytes_in_buffer = len;
	src->pub.next_input_byte = (JOCTET*)data;
}


/*** RGB -> JPEG on mem ***/

METHODDEF(void) memory_init_destination (j_compress_ptr cinfo){

	memory_dst_ptr dest;

	dest = (memory_dst_ptr)cinfo->dest;

	dest->pub.free_in_buffer = dest->buffer_length;

	dest->pub.next_output_byte = dest->buffer;

}

METHODDEF(boolean) memory_empty_output_buffer (j_compress_ptr cinfo){

	return TRUE;

}

METHODDEF(void) memory_term_destination (j_compress_ptr cinfo){

}

GLOBAL(void) jpeg_memory_dst (j_compress_ptr cinfo, void* data, unsigned long len){

	memory_dst_ptr dest;

	if (cinfo->dest == NULL) {/* first time for this JPEG object? */

		cinfo->dest = (struct jpeg_destination_mgr *)
			(*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
			sizeof(memory_destination_mgr));

	}

	dest = (memory_dst_ptr) cinfo->dest;

	dest->pub.init_destination = memory_init_destination;

	dest->pub.empty_output_buffer = memory_empty_output_buffer;

	dest->pub.term_destination = memory_term_destination;

	dest->buffer = (JOCTET *)data;

	dest->buffer_length = len;

}

int jpeg_encode(const unsigned char * const src,
	const int width, const int height, const int channel,
	const unsigned char * const dst,
	const int quality, const int dct_mode, const bool isOptimize)
{
	struct	jpeg_compress_struct	cinfo;
	struct	jpeg_error_mgr	jerr;
	int ret;

	cinfo.err = jpeg_std_error( &jerr );

	jpeg_create_compress( &cinfo );

	jpeg_memory_dst(&cinfo, (void *)dst, width * height * channel);
	//jpeg_stdio_dest( &cinfo, outfile );

	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = channel;

	if(channel==3)
		cinfo.in_color_space = JCS_EXT_BGR;
	else if(channel==1)
		cinfo.in_color_space = JCS_GRAYSCALE;

	jpeg_set_defaults( &cinfo );
	jpeg_set_quality( &cinfo, quality, TRUE );

	//if(isOptimize)cinfo.optimize_coding=TRUE;
	if(isOptimize)cinfo.arith_code = TRUE;

	//	  
	cinfo.dct_method=(J_DCT_METHOD)dct_mode;

	JSAMPROW	row_pointer[ 1 ];

	jpeg_start_compress( &cinfo, TRUE );
	const int row_stride = width * channel;

	uchar* s = (uchar*)src;
	while ( cinfo.next_scanline < height)
	{
		row_pointer[ 0 ] = (JSAMPLE *)s;

		jpeg_write_scanlines( &cinfo, row_pointer, 1 );

		s += row_stride;
	}

	jpeg_finish_compress( &cinfo );
	ret = width * height * channel - cinfo.dest->free_in_buffer;
	jpeg_destroy_compress( &cinfo );

	return ret;
}


bool dataIsValidJPEG(uchar* data, int size)
{
	const char *bytes = (const char*)data;

	return (bytes[0] == (char)0xff && 
		bytes[1] == (char)0xd8 &&
		bytes[size-2] == (char)0xff &&
		bytes[size-1] == (char)0xd9);
}

void getHeader(unsigned char * mem_src, int size, int& width, int& height, int& channels)
{
	if(!dataIsValidJPEG(mem_src,size))cout<<"JPEG file has some error"<<endl;
	
	struct	jpeg_decompress_struct	cinfo;
	struct	jpeg_error_mgr	jerr;

	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_decompress( &cinfo );
	jpeg_memory_src(&cinfo, mem_src, sizeof(char)*size);
	jpeg_read_header( &cinfo, FALSE );
	jpeg_start_decompress( &cinfo );
	

	width = cinfo.output_width;
	height = cinfo.output_height;
	channels = cinfo.output_components;
}
int jpeg_decode( unsigned char * mem_src, int size, unsigned char * mem_dst,
	int width, int height )
{
	if(!dataIsValidJPEG(mem_src,size))cout<<"JPEG file has some error"<<endl;
	struct	jpeg_decompress_struct	cinfo;
	struct	jpeg_error_mgr	jerr;

	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_decompress( &cinfo );

	jpeg_memory_src(&cinfo, mem_src, sizeof(char)*size);

	//jpeg_stdio_src( &cinfo, infile );

	int	row_stride;
	JSAMPARRAY	row_buffer;

	jpeg_read_header( &cinfo, TRUE );

	jpeg_start_decompress( &cinfo );

	if(cinfo.output_components == 1)
		cinfo.out_color_space = JCS_GRAYSCALE;
	else
		cinfo.out_color_space = JCS_EXT_BGR;

	//cout<<"channel, space "<<cinfo.output_components<<","<<cinfo.out_color_space<<endl;


	row_stride = cinfo.output_width * cinfo.output_components;
	row_buffer = ( *cinfo.mem->alloc_sarray )
		(( j_common_ptr ) &cinfo, JPOOL_IMAGE, row_stride, 1 );

	width = cinfo.output_width;
	height = cinfo.output_height;
	//image.resize( width * height * 3 );
	
	uchar* dst = mem_dst;
	int step =  width * cinfo.out_color_components;
	while ( cinfo.output_scanline < cinfo.output_height )
	{
		//cout<<cinfo.output_scanline<<endl;
		jpeg_read_scanlines( &cinfo, row_buffer, 1 );
		unsigned char	*src=row_buffer[ 0 ];

		memcpy(dst,src,sizeof(char)*width * cinfo.out_color_components);
		dst+=step;
	}
	// cout<<"finish"<<endl;
	jpeg_finish_decompress( &cinfo );
	//cout<<"destroy"<<endl;
	jpeg_destroy_decompress( &cinfo );

	return 0;
}


int imdecodeJPEG(const unsigned char* buff, int buff_size, Mat& dest)
{
	int w,h,c;
	getHeader((uchar*)buff,buff_size,w,h,c);
	dest = Mat::zeros(Size(w,h),CV_MAKE_TYPE(CV_8U,c));
	jpeg_decode((uchar*)buff,buff_size,dest.data,w,h);
	return 0;
}
int imdecodeJPEG(const vector<uchar>& buff, Mat& dest)
{
	uchar* dst = (uchar*)&buff[0];
	int w,h,c;
	getHeader((uchar*)dst,buff.size(),w,h,c);
	dest = Mat::zeros(Size(w,h),CV_MAKE_TYPE(CV_8U,c));
	jpeg_decode(dst,buff.size(),dest.data,dest.cols,dest.rows);
	return 0;
}

int imencodeJPEG(const Mat& src, std::vector<uchar>& buff, int q, int DCT_MODE, bool isOpt)
{
	buff.clear();
	buff.resize(src.size().area()*src.channels());
	uchar* dst = &buff[0];
	int s = jpeg_encode(src.data, src.cols,src.rows,src.channels(),dst,q,DCT_MODE,isOpt);

	buff.resize(s);
	//buff.clear();
	return s;
}

int imencodeJPEG(const Mat& src, uchar* buff, int q, int DCT_MODE, bool isOpt)
{
	int s = jpeg_encode(src.data, src.cols,src.rows,src.channels(),buff,q,DCT_MODE,isOpt);
	return s;
}

void degradeJPEG(const Mat& src, Mat& dest, int q, int DCT_MODE, bool isOpt, int& size, double& bpp)
{
	vector<uchar> buff;
	imencodeJPEG(src,buff,q,DCT_MODE,isOpt);
	size = buff.size();
	bpp = 8.0 * (double)size/(double)src.size().area();
	imdecodeJPEG(buff,dest);
}