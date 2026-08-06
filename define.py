
APPNAME = "Shade Wave"
VERSION = "2.137.8"

SUPPORTED_FORMATS_RGB = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.heic', '.jxl')
SUPPORTED_FORMATS_RAW = ('.cr2', '.cr3', '.crw', '.nef', '.nrw', '.arw', '.dng', '.orf', '.raf', '.rw2', '.sr2', '.pef', '.raw', '.3fr', '.fff', '.erf', '.kdc', '.dcr', '.mrw', '.rwl', '.srw', '.mef')
# EXR(HDR リニア)は pyvips 非対応で OpenEXR 経由。RGB とは別経路で扱うので独立タプルにする。
SUPPORTED_FORMATS_EXR = ('.exr',)
