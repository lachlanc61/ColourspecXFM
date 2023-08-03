output_dir = "/home/lachlan/CODEBASE/xfmkit/data/processed_maps/carlos_full"
import os, sys
import numpy as np
import csv

import xfmkit.utils as utils

output_dir = "/home/lachlan/CODEBASE/xfmkit/data/processed_maps/carlos_full"


def export_regions(categories, dimensions, output_directory='.'):
    """
    export each category as a region file in geopixe-format csv

    expects y axis inverted
    """

    n_categories, category_list = utils.count_categories(categories)

    for i in range(n_categories):
        
        #flip y axis
        categories_ = utils.map_roll(categories, dimensions)
        categories_ = np.flip(categories_, axis=0)
        categories_, ___ = utils.map_unroll(categories_)

        filtered_ = np.where(categories_ == i)[0]

        #write it
        print(f"writing {i}")
        write_region(filtered_, i, dimensions, output_directory)

    return

def write_region(region, index: int, dimensions, output_directory='.'):
    """
    generate a region file matching geopixe expected format

    highly nonstandard csv format

    header:
    - large stock header block
    - header fields as single-row csv WITHOUT trailing commas
    
    data:
    - first row of first column prepended with "Q"
    - 100 columns per row, WITH trailing commas
    - last row truncated to N%100, WITHOUT trailing comma
    """

    tempfile = os.path.join(output_directory, f"regions_xfmkit-region{index}.tmp.csv")
    outfile = os.path.join(output_directory, f"regions_xfmkit-region{index}.csv")    

    region_ = region.tolist()
    region_.insert(0, "Q")

    result = []

    i=0
    length=0

    #sort into rows of 100 columns, with overflow in last row
    while i*100+100 <= len(region_):
        row_ = region_[i*100:i*100+100]
        result.append(row_)
        length+=(len(row_))
        i+=1

    row_=region_[i*100::]
    result.append(row_)
    length+=(len(row_))

    #several cycles of tempfile->outfile to deal with line ending weirdness, very hacky

    #   write the header to a tempfile, then strip carriage returns and write header to output
    with open(tempfile, 'w', newline='') as ftemp:
        write_region_header(ftemp, dimensions)  

    with open(tempfile, 'r', newline='') as ftemp:
        with open(outfile, 'w', newline='') as fout:
            lines=ftemp.readlines()
            for line in lines:
                line = line.rstrip('\r\n')
                print(line, file=fout)

    #   write data to fresh tempfile then append to outfile to get the line endings the way geopixe expects
    with open(tempfile, 'w', newline='') as ftemp:
        writer = csv.writer(ftemp)
        writer.writerows(result)

    with open(tempfile, 'r', newline='') as ftemp:
        with open(outfile, 'a', newline='') as fout:
            lines=ftemp.readlines()
            last = lines[-1]
            for line in lines:
                if not line is last:
                    line = line.rstrip('\r\n') + ','
                    print(line, file=fout)
                else:
                    line = line.rstrip('\r\n')
                    print(line, file=fout)

    #remove the tempfile
    os.remove(tempfile)

    return

def write_region_header(f, dimensions):
    """
    write geopixe expected header to opened file f
    """

    offset_line = [[ 'Offset', 0, 0 ]]
    compress_line = [[ 'Compress', 1, 1 ]]
    image_line = [[ 'Image', dimensions[1], dimensions[0] ]]

    print(image_line)

    print(type(image_line))

    f.write(REGION_HEADER)
    writer = csv.writer(f)

    writer.writerows(image_line)
    writer.writerows(offset_line)
    writer.writerows(compress_line)

    return



#header block from geopixe documentation
REGION_HEADER = """# GeoPIXE region export file, saved by "OnButton_Image_Table_Export_Regions".
#
# Saves pixel selection as "Q" vector of indices into the image array.
# This is for current image size given by "Image", which may be some "Offset" into
# full image ("Original"). Additionally, both the image and original may have been
# "Compressed" in X or Y during sorting from event data.
# These parameters become important to import correctly back into GeoPIXE, to translate
# "Q", if image parameters in GeoPIXE differ from those relating to the import "Q".
#
# A minimum set for the "Import" of a region are: Image size ("Image" line), X,Y offsets
# ("Offset" line) and the compression ("Compress" line). Typically, add a "Note" too.
# You can set "Display", if you would like to tag elements as relevant to a region.
# All other paramters are optional on Import. Typically, use "Update: All" after import,
# which applies the imported "Q" to the current images, which determines other data.
#
# Legend:
#    Files: region and DAI image file names
#    Compress: image was compressed by integer factor in X and/or Y during sort from events
#    Original: original full pixel size of image after compress
#    Offset: offset of image within full pixel range of "Original" image (windowed sort)
#    Image: current pixel size of image (after offset and compress)
#    Scan: size of image in mm
#    Scaled: a manual floating scale factor was applied to X/Y (old, not used now)
#    Sample: sample name
#    Grain: grain or analysis point/grain
#    Comment: comment added at data acquisition time
#    Note: note added to region
#    Dwell: dwell time per pixel (ms)
#    Array: 1 (detector array), 0 (single detector)
#    Active: detector channel #s used for image
#    Charge: charge (uC equivalent), flux value
#    n_elements: number of element planes including background, dwell, flux, etc.
#    Element: names of "Element" planes
#    Conc: average concentration (ppm) in selected pixels (includes number of pixels)
#    Error: uncertainty (1-sigma) of concentration (ppm) in selected pixels
#    MDL: minimum detection limit (99% confidence) for selected pixels
#    CentroidX: X centroid conc-weighted across selected pixels
#    CentroidY: Y centroid conc-weighted across selected pixels
#    Mode: 0 (shape region on element image), 1 (Association element-element correlation)
#        Note: All regions become Mode=1 on import as shapes are lost.
#    Display: element displayed to set shape or element-element pair for Association
#    Q: indices of selected pixels in compressed, offset image (continues on multiple lines to EOF)
#
"""