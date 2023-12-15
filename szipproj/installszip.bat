mkdir szipbin
cd szipbin
mkdir lib
mkdir dll
mkdir include
cd ..
copy szip\all\lib\release\szlib.lib szipbin\lib
copy szip\all\libdll\release\szlibdll.dll szipbin\dll
copy szip\all\libdll\release\szlibdll.lib szipbin\dll
copy szip\src\szlib.h szipbin\include
copy szip\src\szip_adpt.h szipbin\include
copy szip\src\ricehdf.h szipbin\include
copy szip\src\SZconfig.h szipbin\include