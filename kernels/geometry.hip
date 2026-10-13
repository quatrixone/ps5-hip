#include <hip/hip_runtime.h>

// Explicit host pitches and fixed measured block sizes bound every store even
// when an observed builtin is wrong. Every observed dimension is recorded.
template<unsigned BX, unsigned BY, unsigned BZ, unsigned GX, unsigned GY, unsigned GZ,
         unsigned LX, unsigned LY, unsigned LZ>
__device__ __forceinline__ void record_geometry(const unsigned *input, unsigned *records,
    unsigned *tail, unsigned pitch_x, unsigned pitch_y, unsigned pitch_z, unsigned record_words) {
    const unsigned bx=blockIdx.x, by=blockIdx.y, bz=blockIdx.z;
    const unsigned tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    if (pitch_x!=BX*GX || pitch_y!=BY*GY || pitch_z!=BZ*GZ ||
        record_words!=BX*GX*BY*GY*BZ*GZ*16 ||
        bx>=GX || by>=GY || bz>=GZ || tx>=BX || ty>=BY || tz>=BZ) return;
    const unsigned x=bx*BX+tx, y=by*BY+ty, z=bz*BZ+tz;
    const unsigned i=(z*pitch_y+y)*pitch_x+x;
    if (i>=pitch_x*pitch_y*pitch_z || i>=record_words/16) return;
    unsigned *p=records+16*i;
    p[0]=bx; p[1]=by; p[2]=bz; p[3]=tx; p[4]=ty; p[5]=tz;
    p[6]=blockDim.x; p[7]=blockDim.y; p[8]=blockDim.z;
    p[9]=gridDim.x; p[10]=gridDim.y; p[11]=gridDim.z;
    p[12]=x; p[13]=y; p[14]=z; p[15]=input[i];
    if (x<LX && y<LY && z<LZ) tail[i]=input[i]^(x*17u+y*257u+z*65537u+0x53a971bdu);
}
extern "C" __global__ __launch_bounds__(64) void geometry_2d(const unsigned *input,
    unsigned *records, unsigned *tail, unsigned pitch_x, unsigned pitch_y,
    unsigned pitch_z, unsigned record_words) {
    record_geometry<8,8,1,3,2,1,21,13,1>(input,records,tail,pitch_x,pitch_y,pitch_z,record_words);
}
extern "C" __global__ __launch_bounds__(64) void geometry_3d(const unsigned *input,
    unsigned *records, unsigned *tail, unsigned pitch_x, unsigned pitch_y,
    unsigned pitch_z, unsigned record_words) {
    record_geometry<8,4,2,2,3,4,13,9,7>(input,records,tail,pitch_x,pitch_y,pitch_z,record_words);
}
