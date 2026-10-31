#!/usr/bin/env python

"""
Description : A set of tools to visualize or process ASP outputs.

Author : Amaury Dehecq
"""

import os
import re
import struct
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geoutils import raster as geor


import os
from subprocess import run

import numpy as np

# Color for printing commands to screen, default is green
cmd_color = "\033[32m"


def run_subprocess_simple(cmd, verbose=True, dryrun=False):
    """
    Small tool to run a command using subprocess.
    Adds a print statement and a verbose option.

    :param cmd: Command to be run with subprocess. Can be a string or a list of strings (to be run with subprocess.run option shell=False)
    :type cmd: list, string
    :param verbose: Set to False to silent the command
    :type verbose: bool
    :param dryrun: If set to True, will only print the command on screen without executing
    :type dryrun: bool
    """
    # Redirect output to null file if verbose is False
    if verbose is False:
        f = open("/dev/null", "w")
        stdout = f
    else:
        stdout = None

    # Print the command with set color and a preceding + symbol
    if type(cmd) is str:
        print(f"{cmd_color:s}+{cmd:s}\033[0m")
    else:
        print("{:s}+{:s}\033[0m".format(cmd_color, " ".join(cmd)))
    if not dryrun:
        # cmd must be a list of all separate arguments
        try:
            run(cmd, stdout=stdout, shell=False, check=False)
        # in case cmd is a string containing all arguments
        except:
            run(cmd, stdout=stdout, shell=True, check=False)


def depth(L):
    """Simple function to check the depth of a nested list"""
    return isinstance(L, list) and max(map(depth, L)) + 1


def check_args_inout(args):
    """
    Accepted arguments for input/outputs are string (in case a single input/output), or some kind of list (list, tuple, numpy array). Convert the former into a list.
    """
    # To comply with both Python3 and 2, string must be detected first
    # in Python3, string have __iter__ attribute too
    if isinstance(args, str):
        if len(args) > 0:  # exclude empty string
            args = [args]
    elif hasattr(args, "__iter__"):  # should work for list, tuples, numpy arrays
        pass
    else:
        raise ValueError("Argument must be iterable (list, tuple, array) or string. Currently set to:")
        print(args)

    return args


def run_subprocess(
    cmds, inputs=[], outputs=[], soft_inputs=[], title=None, verbose=True, dryrun=False, force=False, shell=True
):
    """
    Run a series or commands with subprocess.

    Only run the commands if outputs do not exist or are older than inputs.

    :param cmds: Command(s) to run with subprocess. cmds can be a string or a list of strings (with shell=True), a list of separate arguments without spaces or a list of lists (if shell=False).
    :type cmds: list
    :param inputs: List of inputs
    :param outputs: List of outputs
    :param soft_inputs: List of inputs, which if modified do not force re-executing the command.
    :param title: String to be printed on screen before the command
    :type title: str
    :param force: If set to True, will always run the commands
    :type force: Bool
    :param verbose: if set to False, will not print the commands outputs to screen
    :type verbose: Bool
    :param dryrun: If set to True, will only print the commands without executing them
    :type dryrun: Bool
    :param shell: if set to False, cmds must be a list of arguments without spaces, otherwise, must be a string or list of strings.
    :type as_list: Bool
    :returns: None
    """
    # Convert potential lists into string with space separator
    outputs = check_args_inout(outputs)
    inputs = check_args_inout(inputs)

    # In case as_list is True, check that list cmds is depth 2 (i.e. list of lists)
    cmd_depth = depth(cmds)
    if not shell:
        if cmd_depth == 1:
            if cmds[0].__contains__(" "):
                raise ValueError("Commands contain empty spaces, use option as_list=False instead")
            cmds = [cmds]
        elif cmd_depth == 2:
            pass
        else:
            raise ValueError("cmds must be a list, or list of lists")

    # Otherwise, must be string or list of strings
    else:
        if cmd_depth == 1:
            if not type(cmds[0]) is str:
                raise ValueError("cmds must be a list of str")
        else:
            if not type(cmds) is str:
                raise ValueError("cmds must be a str or list of str")
            cmds = [cmds]

    # Check if all inputs exist
    all_inputs = [*inputs, *soft_inputs]
    if len(all_inputs) > 0:
        for inp in all_inputs:
            if (not os.path.exists(inp)) & (not dryrun):
                raise FileNotFoundError(f"Missing input {inp:s}")

    # Check if output already exist
    if len(outputs) == 0:
        output_exists = False
    else:
        output_exists = True
        older_output_date = np.inf
        for outp in outputs:
            if not os.path.exists(outp):
                output_exists = False
            else:
                older_output_date = min(older_output_date, os.path.getmtime(outp))

    # Check if inputs newer than outputs
    newer = False
    if output_exists:
        for inp in inputs:
            if os.path.getmtime(inp) > older_output_date:
                newer = True

    # Run commands if needed
    if (not output_exists) or newer or force:
        # Print command's title
        if title is not None:
            print(title)

        for cmd in cmds:
            run_subprocess_simple(cmd, verbose=verbose, dryrun=dryrun)
    else:
        print("Nothing to be done")


def dem_coregistration(
    dem_ref: str,
    dem2coreg: str,
    outDEM: str = None,
    tmpPrefix: str = None,
    pc_opts: str = "--save-transformed-source-points --alignment-method point-to-plane --max-displacement 100",
    proj: str = "same",
    extent: str = "same",
    res: str = "same",
    nthreads: int = 0,
    clean: bool = False,
    asp_path: str = None,
    **kwargs,
):
    """
    Coregister two DEMs using ASP tools.

    Inputs:
    :param dem_ref: path to the reference DEM
    :param dem2coreg: path to the DEM to coregister
    :param outDEM: path to the output DEM. Output transform will have same name with suffix '_transform.txt'. Default is same name as input DEM with suffix _coreg.
    :param tmpPrefix: prefix to be added to output directory (same as outDEM) for the temporary files created by ASP. Default is temp_pc/tmp.
    :param pc_opts: optional arguments for pc_align
    :param proj: the output DEM projection - either 'same' (same as dem2coreg), 'default' (will be esimated by ASP point2dem), or a PROJ4 string (Default is 'same')
    :param extent: the output DEM extent - either 'same' (same as dem2coreg), 'default' (will be esimated by ASP point2dem), or a quadruplet of coordinates separated by space (default is 'same')
    :param res: the output DEM resolution - either 'same' (same as dem2coreg), 'default' (will be esimated by ASP point2dem), or a pair of x/y resolution separated by space (default is 'same')
    :param nthreads: number of multi-threaded processes (default is set by ASP)
    :param clean: if set to True, the intermediate files will be removed (Default is True).
    :param **kwargs: optional arguments to be passed to workflow_step (dryrun, verbose...)
    """

    # Get input DEM metadata
    dem2coreg_obj = geor.Raster(dem2coreg, load_data=False)
    proj4 = dem2coreg_obj.crs.to_proj4()
    a0, b0, c0, d0 = (
        dem2coreg_obj.bounds.left,
        dem2coreg_obj.bounds.right,
        dem2coreg_obj.bounds.bottom,
        dem2coreg_obj.bounds.top,
    )
    res0 = dem2coreg_obj.res[0]

    # Set output DEM projection
    p2dem_suffix = ""
    if res == "same":
        p2dem_suffix += "--tr %g" % res0
    elif res == "default":
        pass
    else:
        p2dem_suffix += "--tr %g" % res

    if proj == "same":
        p2dem_suffix += " --t_srs '%s'" % proj4
    elif proj == "default":
        pass
    else:
        p2dem_suffix += " --t_srs '%s'" % proj

    if extent == "same":
        # Extent given in point2dem is pixel center not real extent
        a = a0 + res0 / 2
        b = b0 - res0 / 2
        c = c0 + res0 / 2
        d = d0 - res0 / 2
        p2dem_suffix += f" --t_projwin {a:g} {c:g} {b:g} {d:g}"
    elif extent == "default":
        pass
    else:
        a, c, b, d = np.float32(" ".split(extent))
        p2dem_suffix += f"--t_projwin {a:g} {c:g} {b:g} {d:g}"

    # -- Output files -- #
    # Default output DEM file
    prefix, suffix = os.path.splitext(dem2coreg)
    if outDEM == "default":
        outDEM = f"{prefix}_coreg{suffix}"

    # Output transformation is saved with same prefix as output DEM, and suffix '_transform.txt'
    prefix, suffix = os.path.splitext(outDEM)
    outTrans = prefix + "_transform.txt"

    # Make sure output folder exists
    outDir = os.path.dirname(os.path.abspath(outDEM))
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Output prefix for temporary ASP files
    if tmpPrefix is None:
        tmpPrefix = "temp_pc/tmp"
    pcPrefix = os.path.join(outDir, tmpPrefix)

    # -- Run the coregistration -- #
    # Compute the transformation between the two DEMs
    cmd1 = "%spc_align %s %s -o %s --save-transformed-source-points %s --threads %i" % (
        asp_path, 
        dem_ref,
        dem2coreg,
        pcPrefix,
        pc_opts,
        nthreads,
    )

    # Convert the generated PC to a DEM
    cmd2 = "%spoint2dem %s %s-trans_source.tif --threads %i" % (asp_path, p2dem_suffix, pcPrefix, nthreads)

    # Rename outputs and clean
    if clean:
        cmd3 = "mv {}-trans_source-DEM.tif {}; mv {}-transform.txt {}; rm -r {}-*; rmdir {}".format(
            pcPrefix,
            outDEM,
            pcPrefix,
            outTrans,
            pcPrefix,
            os.path.dirname(pcPrefix),
        )
    else:
        cmd3 = f"mv {pcPrefix}-trans_source-DEM.tif {outDEM}; mv {pcPrefix}-transform.txt {outTrans}"

    run_subprocess(
        [cmd1, cmd2, cmd3],
        [dem_ref, dem2coreg],
        [outDEM, outTrans],
        **kwargs,
    )


def plot_diff(dem1_fn, dem2_fn, cmap="coolwarm", vmin=-30, vmax=30):
    """
    Plot DEM difference map.
    """
    dem1 = geor.Raster(dem1_fn)
    dem2 = geor.Raster(dem2_fn)
    diff = dem2.reproject(dem1) - dem1

    plt.figure()
    diff.plot(cmap=cmap, vmin=vmin, vmax=vmax, cbar_title="Elevation difference (m)")
    plt.show()


def plot_ba_outputs(
    ba_file: str,
    outfile: str = None,
    extent: list = [],
    cmap: str = "Reds",
    vmin: float = 0,
    vmax: float = None,
    size: float = 0.5,
    title: str = None,
):
    """
    Plot the residual error of a bundle adjust file as a scatter plot.

    :param ba_file: path to output file of ASP's bundle_adjust
    :type ba_file: str
    :param outfile: path to output figure. If set to None (default), display on screen.
    :type: str
    :param cmap: Matploblib color map, either as string or as colormap object
    :type: str
    :param vmin: Minimum color map value (Default is 0)
    :param vmax: Maximum color map value (Default is 98th percentile)
    """

    # Check input list
    if len(extent) != 0:
        assert type(extent) in [list, tuple], "extent must be a list or tuple"
        assert len(extent) == 4, "extent must be of length 4"
        extent = np.float32(extent)  # make sure list is float
    else:
        extent = None

    # Read file
    lon, lat, err = np.loadtxt(ba_file, skiprows=2, usecols=(0, 1, 3), delimiter=",", unpack=True)

    # Set default vmax
    if vmax is None:
        vmax = np.percentile(err, 95)

    # #spacing=4000
    # if spacing>0:
    #     x = np.linspace(np.min(lon), np.max(lon), spacing)
    #     y = np.linspace(np.min(lat), np.max(lat), spacing)
    #     vmax = np.percentile(err,95)

    #     # Method 1 - does not work with empty bins
    #     # from scipy.stats import binned_statistic_2d
    #     # erri, _, _, _ = binned_statistic_2d(lon, lat, err, 'mean', bins=[x,y])
    #     # plt.imshow(erri,cmap='Reds',vmin=0,vmax=vmax)
    #     # plt.show()

    #     # Method 2
    #     from scipy import interpolate
    #     X,Y = np.meshgrid(x, y)
    #     Z = interpolate.griddata((lon, lat), err, (X,Y), method='cubic')
    #     plt.imshow(np.flipud(Z), cmap='Reds', vmin=0,vmax=vmax, extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    #     plt.show()

    # Plot the error map
    ax = plt.subplot(111)
    plt.scatter(lon, lat, c=err, cmap=cmap, vmin=vmin, vmax=vmax, s=size, edgecolor="none")
    cb = plt.colorbar()
    cb.set_label("Error")

    # Set title
    if title is not None:
        plt.title(title)

    # Set map extent
    if extent is not None:
        plt.xlim(extent[0], extent[2])
        plt.ylim(extent[1], extent[3])

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, dpi=150)
        print(f"Saved figure {outfile:s}")
        plt.close()


def _read_ip_record(mf):
    """
    Read one IP record from the binary match file.
    Information comtained are x, y, xi, yi, orientation, scale, interest, polarity, octave, scale_lvl, desc
    (Oleg/Scott to explain?)
    Input: - mf, file handle to the in put binary file (in 'rb' mode)
    Output: - iprec, array containing the IP record
    """
    x, y = np.frombuffer(mf.read(8), dtype=np.float32)
    xi, yi = np.frombuffer(mf.read(8), dtype=np.int32)
    orientation, scale, interest = np.frombuffer(mf.read(12), dtype=np.float32)
    (polarity,) = np.frombuffer(mf.read(1), dtype=np.int8)  # or np.bool?
    octave, scale_lvl = np.frombuffer(mf.read(8), dtype=np.uint32)
    (ndesc,) = np.frombuffer(mf.read(8), dtype=np.uint64)
    desc = np.frombuffer(mf.read(int(ndesc * 4)), dtype=np.float32)
    iprec = [x, y, xi, yi, orientation, scale, interest, polarity, octave, scale_lvl, ndesc]
    iprec.extend(desc)
    return iprec


def _write_ip_record(out, iprec):
    """
    Just the reverse operation of _read_ip_record.
    Inputs:
    - out: file handle to the output binary file (in 'wb' mode)
    - iprec: 1D array containing one IP record
    """
    out.write(struct.pack("f", iprec[0]))  # x, y
    out.write(struct.pack("f", iprec[1]))
    out.write(struct.pack("i", int(iprec[2])))  # xi, yi
    out.write(struct.pack("i", int(iprec[3])))
    out.write(struct.pack("f", iprec[4]))  # orientation, scale, interest
    out.write(struct.pack("f", iprec[5]))
    out.write(struct.pack("f", iprec[6]))
    out.write(struct.pack("?", iprec[7]))  # polarity  # use fmt ? instead?
    out.write(struct.pack("I", int(iprec[8])))  # octave, scale_lvl
    out.write(struct.pack("I", int(iprec[9])))
    out.write(struct.pack("Q", int(iprec[10])))  # ndesc
    ndesc = int(iprec[10])
    for k in range(ndesc):
        out.write(struct.pack("f", iprec[11 + k]))  # desc
    return


def read_match_file(match_file):
    """
    Read a full binary match file. First two 8-bits contain the number of IPs in each image. Then contains the record for each IP, image1 first, then image2.
    Input:
    - match_file: str, path to the match file
    Outputs:
    - two arrays, containing the IP records for image1 and image2.
    """
    # Open binary file in read mode
    mf = open(match_file, "rb")

    # Read record length
    size1 = np.frombuffer(mf.read(8), dtype=np.uint64)[0]
    size2 = np.frombuffer(mf.read(8), dtype=np.uint64)[0]

    # Read record for each image
    im1_ip = [_read_ip_record(mf) for i in range(size1)]
    im2_ip = [_read_ip_record(mf) for i in range(size2)]

    # Close file
    mf.close()

    return im1_ip, im2_ip


def read_vwip_file(vwip_file):
    """
    Read a full binary interest point file (.vwip). First 8-bits contain the number of IPs in image. Then contains the record for each IP.
    Input:
    - vwip_file: str, path to the IP file
    Outputs:
    - array containing the IP records
    """

    # Open binary file in read mode
    mf = open(vwip_file, "rb")

    # Read record length
    size = np.frombuffer(mf.read(8), dtype=np.uint64)[0]

    # Read all IPs
    im_ip = [_read_ip_record(mf) for i in range(size)]

    # Close file
    mf.close()

    return im_ip


def write_match_file(outfile, im1_ip, im2_ip):
    """
    Write the full binary match file.
    Inputs:
    - outfile: str, path to the output match file
    - im1_ip: array containing all the records for image1
    - im2_ip: array containing all the records for image2
    """

    # Open binary file in write mode
    out = open(outfile, "wb")

    # Read records lengths
    size1 = len(im1_ip)
    size2 = len(im2_ip)

    # Write record length
    out.write(struct.pack("q", size1))
    out.write(struct.pack("q", size2))

    # Write records for both images
    for k in range(size1):
        _write_ip_record(out, im1_ip[k])
    for k in range(size2):
        _write_ip_record(out, im2_ip[k])
    return


def plot_match_file(match_file: str, img: list = None, outfile: str = None, sub: int = 4):
    # Check input img list
    if img is not None:
        assert len(img) == 2, "img must be of length 2"

    ## Read the match file ##
    # One record for each image
    ipL, ipR = read_match_file(match_file)
    ipL = np.array(ipL)
    ipR = np.array(ipR)

    # Downsample the number of matches if too large
    nmax = 5000
    nIPs = len(ipL)
    if nIPs > nmax:
        step = len(ipL) // nmax
        ipL = ipL[::step]
        ipR = ipR[::step]
        print("Found %i matches - plotting only %i" % (nIPs, len(ipL)))
    else:
        print("Found %i matches" % nIPs)

    ## Read the images ##

    if img is None:
        bkg = False
    else:
        imgL = geor.Raster(img[0], downsample=sub, bands=1)  # second band is mask
        imgR = geor.Raster(img[1], downsample=sub, bands=1)
        nyL, nxL = imgL._disk_shape[1:]
        nyR, nxR = imgR._disk_shape[1:]
        bkg = True

    ## Figure ##

    plt.figure(figsize=(15, 4))

    plt.subplot(121)
    if bkg:
        plt.imshow(imgL.data, extent=(0, nxL, nyL, 0), cmap="gray", interpolation="nearest")
    plt.scatter(ipL[:, 0], ipL[:, 1], color="r", marker="o", facecolor="none", s=10)
    plt.title(f"Left matches/image ({nIPs:d} matches)")

    plt.subplot(122)
    if bkg:
        plt.imshow(imgR.data, extent=(0, nxR, nyR, 0), cmap="gray", interpolation="nearest")
    plt.scatter(ipR[:, 0], ipR[:, 1], color="r", marker="o", facecolor="none", s=10)
    plt.title(f"Right matches/image ({nIPs:d} matches)")

    plt.tight_layout()

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, dpi=100)
        print(f"Saved figure {outfile:s}")
        plt.close()


def plot_vwip_file(vwip_file: str, img: str = None, outfile: str = None, sub: int = 4):
    ## Read the IP file ##
    ips = read_vwip_file(vwip_file)
    ips = np.array(ips)

    # Downsample the number of IPs if too large
    nmax = 40000
    if len(ips) > nmax:
        step = len(ips) // nmax
        print("Found %i IPs - plotting only %i" % (len(ips), nmax))
    else:
        step = 1
        print("Found %i matches" % len(ips))

    ## Read the image ##

    if img is None:
        bkg = False
    else:
        rst = geor.Raster(img, downsample=sub, bands=1)  # Load first band only
        ny, nx = rst._disk_shape[1:]
        bkg = True

    ## Figure ##

    plt.figure(figsize=(8, 4))

    if bkg:
        plt.imshow(rst.data, extent=(0, nx, ny, 0), cmap="gray", interpolation="nearest")
    plt.scatter(ips[::step, 0], ips[::step, 1], color="r", marker="o", facecolor="none", s=5)
    plt.title(f"{len(ips):d} image IPs")

    plt.tight_layout()

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, dpi=100)
        print(f"Saved figure {outfile:s}")
        plt.close()


def cnet_to_matches(cnet_file: str, outPrefix: str, refDEM: str = None, dh_max: float = 50):
    """
    Read a cnet file saved as csv by bundle_adjust and save as ASP's match file.
    Optionally filter points based on elevation difference with a reference DEM.

    :param cnet_file: str, path to the cnet.csv file
    :param out_match_file: str, path to the output match file
    :param refDEM: str, path to a reference DEM. Default is None.
    :param dh_max: float, maximum elevation difference w.r.t the reference DEM. Default is 50 m.
    """
    # Read the cnet file
    # uses pandas because np.genfromtxt does not handle missing columns well
    # Force reading 22 columns even if first line has missing columns
    data = np.array(
        pd.read_csv(cnet_file, sep=" ", header=None, names=np.arange(23), comment="#")
    )  # np.array(pd.read_csv(cnet_file, sep=' '), names=np.arange(22))
    lat = data[:, 1]
    lon = data[:, 2]
    z = data[:, 3]

    if refDEM is not None:
        # Read reference elevation at known points
        # Use interp_points as reduce_points has a bug for out-of-bounds when data loaded
        # See https://github.com/GlacioHack/geoutils/issues/718
        dem = geor.Raster(refDEM, load_data=False)
        # zinterp = dem.reduce_points((lon, lat), input_latlon=True)
        zinterp = dem.interp_points((lat, lon), input_latlon=True)
        zdiff = z - zinterp

        # Points outside of DEM validity (e.g. water bodies) are set to nan
        # Keep only valid points
        with warnings.catch_warnings(action="ignore"):
            valids = np.where(~np.isnan(zinterp) & (np.abs(zdiff) < dh_max))[0]
        dataf = data[valids]
        print(f"Elevation filter - Keeping {len(valids):d}/{len(data):d} valid points")
    else:
        dataf = data

    # Find list of all images matched
    img1 = dataf[:, 7]
    img2 = dataf[:, 12]
    img3 = dataf[:, 17]
    all_images = np.hstack((img1, img2, img3)).astype("str")

    # Remove empty columns
    all_images = all_images[all_images != "nan"]

    # Unique list of images
    images = np.unique(all_images)

    # Loop through all pairs
    for k, p1 in enumerate(images):
        for p2 in images[k + 1 :]:
            # Image IDs
            ID1 = os.path.splitext(os.path.basename(p1))[0]
            ID2 = os.path.splitext(os.path.basename(p2))[0]

            # Find matches for first pair
            inds1 = np.where((img1 == p1) & (img2 == p2))[0]

            if len(inds1) > 0:
                # Find IPs locations
                im1_ip = [[line[8], line[9], int(line[8]), int(line[9]), 0, 0, 0, 0, 0, 0, 0] for line in dataf[inds1]]
                im2_ip = [
                    [line[13], line[14], int(line[13]), int(line[14]), 0, 0, 0, 0, 0, 0, 0] for line in dataf[inds1]
                ]

                # Save to file
                mfile = "".join((outPrefix, "-", ID1, "__", ID2, ".match"))
                print(f"Saving {len(inds1):d} points in file {mfile:s}")
                write_match_file(mfile, im1_ip, im2_ip)

            # Find matches for optional second pair
            inds2 = np.where((img1 == p1) & (img3 == p2))[0]

            if len(inds2) > 0:
                # Find IPs locations
                im1_ip = [[line[8], line[9], int(line[8]), int(line[9]), 0, 0, 0, 0, 0, 0, 0] for line in dataf[inds2]]
                im2_ip = [
                    [line[18], line[19], int(line[18]), int(line[19]), 0, 0, 0, 0, 0, 0, 0] for line in dataf[inds2]
                ]

                # Save to file
                mfile = "".join((outPrefix, "-", ID1, "__", ID2, ".match"))
                print(f"Saving {len(inds2):d} points in file {mfile:s}")
                write_match_file(mfile, im1_ip, im2_ip)


