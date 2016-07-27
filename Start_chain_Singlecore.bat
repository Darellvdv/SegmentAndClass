@echo off

SET "dir=D:\pinapple\multicore"
SET "scriptdir=D:\pinapple\SegmentAndClass\OBIA_chain.py"
SET "batchdir=D:\pinapple\SegmentAndClass\LSMSSegmentation_chain.bat"

for %%X in ("%dir%\*.tif") do Python %scriptdir% --image %%~fX --output %dir%\%%~nX --bat %batchdir% --pre yes --att rgbnir

PAUSE
