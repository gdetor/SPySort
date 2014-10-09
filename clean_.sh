#
#/usr/bin/sh
#
# Clean all *.dat and *.pyc files from SPySort directories

find . -name "*.dat" -type f|xargs rm -f >> /dev/null 2>&1
find . -name "*.pyc" -type f|xargs rm -f >> /dev/null 2>&1
