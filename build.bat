
::call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
echo DevEnvDir set to: %DevEnvDir%

::cl icbc_test.cpp tracy/TracyClient.cpp /O2 /arch:AVX512 Advapi32.lib User32.lib /DTRACY_ENABLE /Zi
cl icbc_test.cpp /O2 /arch:AVX512
