import numpy as np
import requests
import pathlib
import warnings
from PIL import Image
from io import BytesIO
from astropy.table import Table
import os

# suppresses scientific notation, turn on for PS with objID
np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})


def load_cand_coord(path="/tmp/CandCoord.dat", onlynumeric=True):
    # Save using CSV.write, otherwise RA DEC are truncated!
    if onlynumeric:
        coords = np.loadtxt(path, skiprows=1, delimiter=',')
        names, ra, dec = coords[:, 0].astype(int), coords[:, 1], coords[:, 2]
    else:
        coords = np.genfromtxt(path, delimiter=',', dtype='str')
        names, _ra, _dec = coords[:, 0], coords[:, 1], coords[:, 2]
        ra, dec = [], []
        for (_1, _2) in zip(_ra, _dec):
            ra.append(float(_1))
            dec.append(float(_2))
        ra = np.array(ra)
        dec = np.array(dec)
    return names, ra, dec


# https://ps1images.stsci.edu/ps1image.html
def getimages(ra, dec, size=1920, filters="grizy"):
    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table


def geturl(ra,
           dec,
           size=1920,
           output_size=None,
           filters="grizy",
           format="jpg",
           color=False):
    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and format == "fits":
        raise ValueError(
            "color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra, dec, size=size, filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase + filename)
    return url


def getgrayim(qid,
              ra,
              dec,
              size=1920,
              output_size=None,
              filter="g",
              format="jpg",
              path="/tmp/FindingChartsPS"):
    """Get grayscale image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if not os.path.isdir(path):
        os.mkdir(path)

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra,
                 dec,
                 size=size,
                 filters=filter,
                 output_size=output_size,
                 format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    im.save(path + str(qid) + '.jpg')
    os.system(
        "convert {} /home/francio/.dss-overlay-red.gif -gravity center -composite -format png {}.png"
        .format(path + str(qid) + '.jpg', path + str(qid) + '.jpg'))
    return 0


def getcolorim(qid,
               ra,
               dec,
               size=1920,
               output_size=None,
               filters="grizy",
               format="jpg",
               path="/tmp/FindingChartsPS"):
    """Get color image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if not os.path.isdir(path):
        os.mkdir(path)

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra,
                 dec,
                 size=size,
                 filters=filters,
                 output_size=output_size,
                 format=format,
                 color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    im.save(path + str(qid) + '.jpg')
    os.system(
        "convert {} /home/francio/.dss-overlay-red.gif -gravity center -composite -format png {}.png"
        .format(path + str(qid) + '.jpg', path + str(qid) + '.jpg'))
    return 0


def getCandCutout(pathList="/tmp/CandCoord.dat",
                  size=720,
                  outsize=None,
                  filters="grizy",
                  format="jpg",
                  outPath="/tmp/FindingChartsPS/",
                  onlynumeric=True):
    qid, ra, dec = load_cand_coord(path=pathList, onlynumeric=onlynumeric)
    for (_qid, _ra, _dec) in zip(qid, ra, dec):
        print("Fetching qid: {}".format(_qid))
        if _dec > -30.5 and len(filters) == 1:
            getgrayim(_qid,
                      _ra,
                      _dec,
                      path=outPath,
                      size=size,
                      output_size=outsize,
                      filter=filters,
                      format=format)
        elif _dec > -30.5 and len(filters) > 1:
            getcolorim(_qid,
                       _ra,
                       _dec,
                       path=outPath,
                       size=size,
                       output_size=outsize,
                       filters=filters,
                       format=format)
        else:
            print("QID {} is not in the PS footprint!".format(_qid))
    return 0


def get_legacysurvey(id, ra, dec, size=240, scale=0.25, path="/tmp"):  # max size (pixel) = 512
    url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&size={size}&layer=ls-dr10&pixscale={scale}"
    # for fits https://www.legacysurvey.org/viewer/fits-cutout?ra=0.&dec=0.&size=120&layer=ls-dr10&pixscale=120
    print(f"Downloading: {url}\n")
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    im.save(path + "/" + str(id) + ".jpg")
    os.system(
        "convert {} /home/francio/.dss-overlay-red.gif -gravity center -composite -format png {}.png"
        .format(path + "/" + str(id) + '.jpg', path + "/" + str(id) + '.jpg'))
    return 0


def download_fits(ra, dec, name, out_path, size=120, filters="y"):
    # this is in principle a list, does not matter if single filter
    url = geturl(ra, dec, size=size, format='fits', filters=filters,
                 color=False)  # 30 arcsec default for searching lenses, with y filter
    destination = (pathlib.Path(out_path) / name).with_suffix(".fits")

    # get data
    for _ in url:
        res = requests.get(_)
        if res.status_code == 200:  # http 200 means success
            with open(destination, 'wb') as file_handle:  # wb means Write Binary
                file_handle.write(res.content)
        else:
            warnings.warn(f"Could not download fits for {name}")
