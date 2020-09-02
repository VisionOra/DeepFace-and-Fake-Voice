sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
sudo su
# prompt for password
TENSORFLOW_DIR=`python -c "import imp; print(imp.find_module('tensorflow')[1])"`
for addr in 0xC6A93C 0xC6A99C 0xC6A9EC 0xC6AA0C 0xC6AA1C 0xC6AA3C; do printf '\x02' | dd conv=notrunc of=${TENSORFLOW_DIR}/python/_pywrap_tensorflow.so bs=1 seek=$((addr)) ; done

readelf -V ${TENSORFLOW_DIR}/python/_pywrap_tensorflow.so
exit

mkdir ~/.my_stubs
cd ~/.my_stubs
MYSTUBS=~/.my_stubs


printf "#include <time.h>\n#include <string.h>\nvoid* memcpy(void *dest, const void *src, size_t n) {\nreturn memmove(dest, src, n);\n}\nint clock_gettime(clockid_t clk_id, struct timespec *tp) {\nreturn clock_gettime(clk_id, tp);\n}" > mylibc.c
gcc -s -shared -o mylibc.so -fPIC -fno-builtin mylibc.c

printf "#include <functional>\nvoid std::__throw_bad_function_call(void) {\nexit(1);\n}" > bad_function.cc
gcc -std=c++11 -s -shared -o bad_function.so -fPIC -fno-builtin bad_function.cc

git clone https://github.com/gcc-mirror/gcc.git

cd gcc
mkdir my_include
mkdir my_include/ext
cp libstdc++-v3/include/ext/aligned_buffer.h my_include/ext
gcc -I$PWD/my_include -std=c++11 -fpermissive -s -shared -o $MYSTUBS/hashtable.so -fPIC -fno-builtin libstdc++-v3/src/c++11/hashtable_c++0x.cc
gcc -std=c++11 -fpermissive -s -shared -o $MYSTUBS/chrono.so -fPIC -fno-builtin libstdc++-v3/src/c++11/chrono.cc
gcc -std=c++11 -fpermissive -s -shared -o $MYSTUBS/random.so -fPIC -fno-builtin libstdc++-v3/src/c++11/random.cc
gcc -std=c++11 -fpermissive -s -shared -o $MYSTUBS/hash_bytes.so -fPIC -fno-builtin ./libstdc++-v3/libsupc++/hash_bytes.cc