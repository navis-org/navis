#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import datetime
import io
import os
import re
import requests
import tempfile
import tarfile

import multiprocessing as mp
import numpy as np
import pandas as pd

from abc import ABC
from functools import partial, wraps
from pathlib import Path
from typing import List, Union, Iterable, Dict, Optional, Any, IO
from typing_extensions import Literal
from zipfile import ZipFile, ZipInfo
from ftplib import FTP

from .. import config, utils, core

try:
    import zlib
    import zipfile

    compression = zipfile.ZIP_DEFLATED
except ModuleNotFoundError:
    compression = zipfile.ZIP_STORED

__all__ = ["BaseReader"]

# Set up logging
logger = config.get_logger(__name__)

DEFAULT_INCLUDE_SUBDIRS = False

# Regular expression to figure out if a string is a regex pattern
rgx = re.compile(r"[\\\.\?\[\]\+\^\$\*]")


def merge_dicts(*dicts: Optional[Dict], **kwargs) -> Dict:
    """Merge dicts and kwargs left to right.

    Ignores None arguments.
    """
    # handles nones
    out = dict()
    for d in dicts:
        if d:
            out.update(d)
    out.update(kwargs)
    return out


def handle_errors(func):
    """Decorator for read_buffer and read_dataframe methods to handle errors.

    Catches exceptions, logs/raises and potentially return `None`.

    Note: various other BaseReader methods have their own error handling.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    callable
        Wrapped function.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        attrs = kwargs.get("attrs", {})  # we rely on this being a keyword argument!
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            # Check if we can provide any hint as to which file failed
            id = self.name_fallback
            for a in ("file", "origin", "name"):
                if a in attrs:
                    id = attrs[a]
                    break

            if self.errors == "raise":
                raise ReadError(f"Error reading {id}. See above traceback for details.") from e
            elif self.errors == "log":
                logger.exception(f"Failed to read {id}", exc_info=True)

            return None

    return wrapper


class Writer:
    """Writer class that takes care of things like filenames, archives, etc.

    Parameters
    ----------
    write_func :    callable
                    Writer function to write a single file to disk. Must
                    accept a `filepath` parameter.
    ext :           str, optional
                    File extension - e.g. '.swc'.

    """

    def __init__(self, write_func, ext):
        assert callable(write_func)
        if ext:
            assert isinstance(ext, str) and ext.startswith(".")
        self.write_func = write_func
        self.ext = ext

    def write_single(self, x, filepath, **kwargs):
        """Write single object to file."""
        # try to str.format any path-like
        try:
            as_str = os.fspath(filepath)
        except TypeError:
            raise ValueError(
                f'`filepath` must be str or pathlib.Path, got "{type(filepath)}"'
            )

        # Format filename (e.g. "{neuron.name}.swc")
        formatted_str = as_str.format(neuron=x)

        # If it was formatted, make sure it has correct extension
        if self.ext and formatted_str != as_str and not as_str.endswith(self.ext):
            raise ValueError(f"Formattable filepaths must end with '{self.ext}'")

        filepath = Path(formatted_str)

        # Expand user - otherwise .exists() might fail
        filepath = filepath.expanduser()

        # If not specified, generate filename
        if self.ext and not str(filepath).endswith(self.ext):
            filepath = filepath / f"{x.id}{self.ext}"

        # Make sure the parent directory exists
        if not filepath.parent.exists():
            raise ValueError(f"Parent folder {filepath.parent} must exist.")

        # Track the path we put this (and presumably all other files in)
        self.path = Path(filepath)
        while not self.path.is_dir():
            self.path = self.path.parent

        return self.write_func(x, filepath=filepath, **kwargs)

    def write_many(self, x, filepath, **kwargs):
        """Write multiple files to folder."""
        if not utils.is_iterable(filepath):
            # Assume this is a folder if it doesn't end with e.g. '.swc'
            is_filename = str(filepath).endswith(self.ext) if self.ext else False
            is_single = len(x) == 1
            is_formattable = "{" in str(filepath) and "}" in str(filepath)
            if not is_filename or is_single or is_formattable:
                filepath = [filepath] * len(x)
            else:
                raise ValueError(
                    "`filepath` must either be a folder, a "
                    "formattable filepath or a list of filepaths"
                    "when saving multiple neurons."
                )

        if len(filepath) != len(x):
            raise ValueError(
                f"Got {len(filepath)} file names for " f"{len(x)} neurons."
            )

        # At this point filepath is iterable
        filepath: Iterable[str]
        for n, f in config.tqdm(
            zip(x, filepath),
            disable=config.pbar_hide,
            leave=config.pbar_leave,
            total=len(x),
            desc="Writing",
        ):
            self.write_single(n, filepath=f, **kwargs)

    def write_zip(self, x, filepath, **kwargs):
        """Write files to zip."""
        filepath = Path(filepath).expanduser()
        # Parse pattern, if given
        pattern = "{neuron.id}" + (self.ext if self.ext else "")
        if "@" in str(filepath):
            pattern, filename = filepath.name.split("@")
            filepath = filepath.parent / filename

        # Make sure we have an iterable
        x = core.NeuronList(x)

        with ZipFile(filepath, mode="w") as zf:
            # Context-manager will remove temporary directory and its contents
            with tempfile.TemporaryDirectory() as tempdir:
                for n in config.tqdm(
                    x,
                    disable=config.pbar_hide,
                    leave=config.pbar_leave,
                    total=len(x),
                    desc="Writing",
                ):
                    # Save to temporary file
                    f = None
                    try:
                        # Generate temporary filename
                        f = os.path.join(tempdir, pattern.format(neuron=n))
                        # Write to temporary file
                        self.write_single(n, filepath=f, **kwargs)
                        # Add file to zip
                        zf.write(
                            f,
                            arcname=pattern.format(neuron=n),
                            compress_type=compression,
                        )
                    except BaseException:
                        raise
                    finally:
                        # Remove temporary file - we do this inside the loop
                        # to avoid unnecessarily occupying space as we write
                        if f:
                            os.remove(f)

        # Set filepath to zipfile -> this overwrite filepath set in write_single
        # (which would be the temporary file)
        self.path = Path(filepath)

    def write_any(self, x, filepath, **kwargs):
        """Write any to file. Default entry point."""
        # If target is a zipfile
        if isinstance(filepath, (str, Path)) and str(filepath).endswith(".zip"):
            return self.write_zip(x, filepath=filepath, **kwargs)
        elif isinstance(x, core.NeuronList):
            return self.write_many(x, filepath=filepath, **kwargs)
        else:
            return self.write_single(x, filepath=filepath, **kwargs)


class BaseReader(ABC):
    """Abstract reader to parse various inputs into neurons.

    Any subclass should implement at least one of `read_buffer` or
    `read_dataframe`. Entry methods such as `read_any` will pass
    and parse an input through to the appropriate method.

    Parameters
    ----------
    fmt :           str
                    A string describing how to parse filenames into neuron
                    properties. For example '{id}.swc'.
    file_ext :      str | tuple
                    The file extension(s) to look for when searching folders.
                    For example '.swc'. Alternatively, you can re-implement
                    the `is_valid_file` method for more complex filters. That
                    method needs to be able to deal with: Path objects, ZipInfo
                    objects and strings.
    name_fallback : str
                    Fallback for name when reading from e.g. string.
    attrs :         dict
                    Additional attributes to use when creating the neuron.
                    Will be overwritten by later additions (e.g. from `fmt`).
    ignore_hidden : bool
                    Whether to ignore files that start with "._".
    errors :        "raise" | "log" | "ignore"
                    What to do when an error is encountered:
                     - "raise" (default) will raise an error
                     - "log" will log a warning and return `None`
                     - "ignore" will return `None`
                    Applies only to errors in parsing file contents into neurons
                    not to errors in reading files, archives, URLs, etc.

    """

    def __init__(
        self,
        fmt: str,
        file_ext: str,
        name_fallback: str = "NA",
        read_binary: bool = False,
        attrs: Optional[Dict[str, Any]] = None,
        ignore_hidden=True,
        errors="raise"
    ):
        self.attrs = attrs
        self.fmt = fmt
        self.file_ext = file_ext
        self.name_fallback = name_fallback
        self.read_binary = read_binary
        self.ignore_hidden = ignore_hidden
        self.errors = errors

        assert errors in ("raise", "log", "ignore")

    @property
    def file_ext(self):
        return self._file_ext

    @file_ext.setter
    def file_ext(self, value):
        """Makes sure file_ext is always a tuple."""
        if isinstance(value, str):
            value = (value,)

        if any((ext.startswith("*") for ext in value)):
            raise ValueError('File extension must be ".ext", not "*.ext"')

        self._file_ext = value

    def format_output(self, x):
        """Format output into NeuronList.

        Replace this method if output is not (always) a NeuronList.
        See for example NrrdReader in nrrd_io.py.
        """
        if not x:
            return core.NeuronList([])
        else:
            return core.NeuronList([n for n in x if n])

    def files_in_dir(
        self, dpath: Path, include_subdirs: bool = DEFAULT_INCLUDE_SUBDIRS
    ) -> Iterable[Path]:
        """List files to read in directory."""
        if not isinstance(dpath, Path):
            dpath = Path(dpath)

        if "*" in str(dpath):
            pattern = str(dpath.name)
            dpath = dpath.parent
        else:
            pattern = "*"

        dpath = dpath.expanduser()
        if include_subdirs:
            pattern = os.path.join("**", pattern)

        yield from (f for f in dpath.glob(pattern) if self.is_valid_file(f))

    def is_valid_file(self, file):
        """Return true if file should be considered for reading."""
        if isinstance(file, ZipInfo):
            file = file.filename
        elif isinstance(file, tarfile.TarInfo):
            file = file.name
        elif isinstance(file, Path):
            file = file.name

        if self.ignore_hidden and str(file).startswith("._"):
            return False

        for ext in self.file_ext:
            if str(file).endswith(ext):
                return True
        return False

    def _make_attributes(
        self, *dicts: Optional[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """Combine attributes with a timestamp and those defined on the object.

        Later additions take precedence:

        - created_at (now)
        - object-defined attributes
        - additional dicts given as *args
        - additional attributes given as **kwargs

        Returns
        -------
        Dict[str, Any]
            Arbitrary string-keyed attributes.
        """
        return merge_dicts(
            dict(created_at=str(datetime.datetime.now())),
            self.attrs,
            *dicts,
            **kwargs,
        )

    def read_file_path(
        self, fpath: os.PathLike, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.BaseNeuron":
        """Read single file from path into a neuron.

        Parameters
        ----------
        fpath :     str | os.PathLike
                    Path to files.
        attrs :     dict or None
                    Arbitrary attributes to include in the neuron.

        Returns
        -------
        core.BaseNeuron
        """
        p = Path(fpath)
        with open(p, "rb" if self.read_binary else "r") as f:
            props = self.parse_filename(f.name)
            props["origin"] = str(p)
            return self.read_buffer(f, attrs=merge_dicts(props, attrs))

    def read_from_zip(
        self,
        files: Union[str, List[str]],
        zippath: os.PathLike,
        attrs: Optional[Dict[str, Any]] = None,
        on_error: Union[Literal["ignore", Literal["raise"]]] = "ignore",
    ) -> "core.NeuronList":
        """Read given files from a zip into a NeuronList.

        Typically not used directly but via `read_zip()` dispatcher.

        Parameters
        ----------
        files :     zipfile.ZipInfo | list thereof
                    Files inside the ZIP file to read.
        zippath :   str | os.PathLike
                    Path to zip file.
        attrs :     dict or None
                    Arbitrary attributes to include in the TreeNeuron.
        on_error :  'ignore' | 'raise'
                    What do do when error is encountered.

        Returns
        -------
        core.NeuronList

        """
        p = Path(zippath)
        files = utils.make_iterable(files)

        neurons = []
        with ZipFile(p, "r") as zip:
            for file in files:
                # Note the `file` is of type zipfile.ZipInfo here
                props = self.parse_filename(file.orig_filename)
                props["origin"] = str(p)
                try:
                    n = self.read_bytes(zip.read(file), attrs=merge_dicts(props, attrs))
                    neurons.append(n)
                except BaseException:
                    if on_error == "ignore":
                        logger.warning(f'Failed to read "{file.filename}" from zip.')
                    else:
                        raise

        return self.format_output(neurons)

    def read_zip(
        self,
        fpath: os.PathLike,
        parallel="auto",
        limit: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        on_error: Union[Literal["ignore", Literal["raise"]]] = "ignore",
    ) -> "core.NeuronList":
        """Read files from a zip into a NeuronList.

        This is a dispatcher for `.read_from_zip`.

        Parameters
        ----------
        fpath :     str | os.PathLike
                    Path to zip file.
        limit :     int, optional
                    Limit the number of files read from this directory.
        attrs :     dict or None
                    Arbitrary attributes to include in the TreeNeuron.
        on_error :  'ignore' | 'raise'
                    What do do when error is encountered.

        Returns
        -------
        core.NeuronList

        """
        fpath = Path(fpath).expanduser()
        read_fn = partial(
            self.read_from_zip, zippath=fpath, attrs=attrs, on_error=on_error
        )
        neurons = parallel_read_archive(
            read_fn=read_fn,
            fpath=fpath,
            file_ext=self.is_valid_file,
            limit=limit,
            parallel=parallel,
        )
        return self.format_output(neurons)

    def read_from_tar(
        self,
        files: Union[str, List[str]],
        tarpath: os.PathLike,
        attrs: Optional[Dict[str, Any]] = None,
        on_error: Union[Literal["ignore", Literal["raise"]]] = "ignore",
    ) -> "core.NeuronList":
        """Read given files from a tar into a NeuronList.

        Typically not used directly but via `read_tar()` dispatcher.

        Parameters
        ----------
        files :     tarfile.TarInfo | list thereof
                    Files inside the tar file to read.
        tarpath :   str | os.PathLike
                    Path to tar file.
        attrs :     dict or None
                    Arbitrary attributes to include in the TreeNeuron.
        on_error :  'ignore' | 'raise'
                    What do do when error is encountered.

        Returns
        -------
        core.NeuronList

        """
        p = Path(tarpath)
        files = utils.make_iterable(files)

        neurons = []
        with tarfile.open(p, "r") as tf:
            for file in files:
                # Note the `file` is of type tarfile.TarInfo here
                props = self.parse_filename(file.name.split("/")[-1])
                props["origin"] = str(p)
                try:
                    n = self.read_bytes(
                        tf.extractfile(file).read(), attrs=merge_dicts(props, attrs)
                    )
                    neurons.append(n)
                except BaseException:
                    if on_error == "ignore":
                        logger.warning(f'Failed to read "{file.filename}" from tar.')
                    else:
                        raise

        return self.format_output(neurons)

    def read_tar(
        self,
        fpath: os.PathLike,
        parallel="auto",
        limit: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        on_error: Union[Literal["ignore", Literal["raise"]]] = "ignore",
    ) -> "core.NeuronList":
        """Read files from a tar archive into a NeuronList.

        This is a dispatcher for `.read_from_tar`.

        Parameters
        ----------
        fpath :     str | os.PathLike
                    Path to tar file.
        limit :     int, optional
                    Limit the number of files read from this directory.
        attrs :     dict or None
                    Arbitrary attributes to include in the TreeNeuron.
        on_error :  'ignore' | 'raise'
                    What do do when error is encountered.

        Returns
        -------
        core.NeuronList

        """
        fpath = Path(fpath).expanduser()
        read_fn = partial(
            self.read_from_tar, tarpath=fpath, attrs=attrs, on_error=on_error
        )
        neurons = parallel_read_archive(
            read_fn=read_fn,
            fpath=fpath,
            file_ext=self.is_valid_file,
            limit=limit,
            parallel=parallel,
        )
        return self.format_output(neurons)

    def read_ftp(
        self,
        url,
        parallel="auto",
        limit: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        on_error: Union[Literal["ignore", Literal["raise"]]] = "ignore",
    ) -> "core.NeuronList":
        """Read files from an FTP server.

        This is a dispatcher for `.read_from_ftp`.

        Parameters
        ----------
        url :       str
                    Can be the path to a single file or a directory.
        limit :     int, optional
                    Limit the number of files read from this directory.
        attrs :     dict or None
                    Arbitrary attributes to include in the TreeNeuron.
        on_error :  'ignore' | 'raise'
                    What do do when error is encountered.

        Returns
        -------
        core.NeuronList

        """
        # Remove the ftp:// prefix
        url = url.replace("ftp://", "")

        # Split into server and path
        server, path = url.split("/", 1)

        # Check if server contains a port
        if ":" in server:
            server, port = server.split(":")
            port = int(port)
        else:
            port = 21  # default port

        read_fn = partial(self.read_from_ftp, attrs=attrs, on_error=on_error)
        neurons = parallel_read_ftp(
            read_fn=read_fn,
            server=server,
            port=port,
            path=path,
            file_ext=self.is_valid_file,
            limit=limit,
            parallel=parallel,
        )
        return self.format_output(neurons)

    def read_from_ftp(
        self,
        files: Union[str, List[str]],
        ftp: FTP,
        attrs: Optional[Dict[str, Any]] = None,
        on_error: Union[Literal["ignore", Literal["raise"]]] = "ignore",
    ) -> "core.NeuronList":
        """Read given files from an FTP server into a NeuronList.

        Typically not used directly but via `read_ftp()` dispatcher.

        Parameters
        ----------
        files :     tarfile.TarInfo | list thereof
                    Files inside the tar file to read.
        ftp :       ftplib.FTP | "GLOBAL"
                    The FTP client. This should already be connected, logged in
                    and in the correct directory. If "GLOBAL", we will look for a
                    `_FTP` global variable.
        attrs :     dict or None
                    Arbitrary attributes to include in the TreeNeuron.
        on_error :  'ignore' | 'raise'
                    What do do when error is encountered.

        Returns
        -------
        core.NeuronList

        """
        # When reading in parallel, we expect there to be a global FTP connection
        # that was initialized once for each worker process.
        if ftp == "GLOBAL":
            if "_FTP" not in globals():
                raise ValueError("No global FTP connection found.")
            ftp = _FTP

        files = utils.make_iterable(files)

        neurons = []
        for file in files:
            # Read the file into a bytes
            with io.BytesIO() as f:
                ftp.retrbinary("RETR " + file, f.write)
                f.seek(0)
                props = self.parse_filename(file)
                props["origin"] = f"{ftp.host}:{ftp.port}{ftp.pwd()}/{file}"
                try:
                    n = self.read_buffer(f, attrs=merge_dicts(props, attrs))
                    neurons.append(n)
                except BaseException:
                    if on_error == "ignore":
                        logger.warning(f'Failed to read "{file}" from FTP.')
                    else:
                        raise

        return self.format_output(neurons)

    def read_directory(
        self,
        path: os.PathLike,
        include_subdirs=DEFAULT_INCLUDE_SUBDIRS,
        parallel="auto",
        limit: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "core.NeuronList":
        """Read directory of files into a NeuronList.

        Parameters
        ----------
        fpath :             str | os.PathLike
                            Path to directory containing files.
        include_subdirs :   bool, optional
                            Whether to descend into subdirectories, default False.
        parallel :          str | bool | "auto"
        limit :             int, optional
                            Limit the number of files read from this directory.
        attrs :             dict or None
                            Arbitrary attributes to include in the TreeNeurons
                            of the NeuronList

        Returns
        -------
        core.NeuronList

        """
        files = list(self.files_in_dir(Path(path), include_subdirs))

        if isinstance(limit, int):
            files = files[:limit]
        elif isinstance(limit, list):
            files = [f for f in files if f in limit]
        elif isinstance(limit, slice):
            files = files[limit]
        elif isinstance(limit, str):
            # Check if limit is a regex
            if rgx.search(limit):
                files = [f for f in files if re.search(limit, str(f.name))]
            else:
                files = [f for f in files if limit in str(f)]

        read_fn = partial(self.read_file_path, attrs=attrs)
        neurons = parallel_read(read_fn, files, parallel)
        return self.format_output(neurons)

    def read_url(
        self, url: str, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.BaseNeuron":
        """Read file from URL into a neuron.

        Parameters
        ----------
        url :       str
                    URL to file.
        attrs :     dict or None
                    Arbitrary attributes to include in the neuron.

        Returns
        -------
        core.BaseNeuron
        """
        # Note: originally, we used stream=True and passed `r.raw` to the
        # read_buffer function but that caused issue when there was more
        # than one chunk which would require us to concatenate the chunks
        # `via r.raw.iter_content()`.
        # Instead, we will simply read the whole content, wrap it in a BytesIO
        # and pass that to the read_buffer function. This is not ideal as it
        # will load the whole file into memory while the streaming solution
        # may have raised an exception earlier if the file was corrupted or
        # the wrong format.
        with requests.get(url, stream=False) as r:
            r.raise_for_status()
            props = self.parse_filename(url.split("/")[-1])
            props["origin"] = url
            return self.read_buffer(io.BytesIO(r.content), attrs=merge_dicts(props, attrs))

    def read_string(
        self, s: str, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.BaseNeuron":
        """Read single string into a Neuron.

        Parameters
        ----------
        s :         str
                    String.
        attrs :     dict or None
                    Arbitrary attributes to include in the neuron.

        Returns
        -------
        core.BaseNeuron
        """
        sio = io.StringIO(s)
        return self.read_buffer(
            sio, attrs=merge_dicts({"name": self.name_fallback, "origin": "string"}, attrs)
        )

    def read_bytes(
        self, s: str, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.BaseNeuron":
        """Read bytes into a Neuron.

        Parameters
        ----------
        s :         bytes
                    Bytes.
        attrs :     dict or None
                    Arbitrary attributes to include in the neuron.

        Returns
        -------
        core.BaseNeuron
        """
        sio = io.BytesIO(s)
        return self.read_buffer(
            sio, attrs=merge_dicts({"name": self.name_fallback, "origin": "string"}, attrs)
        )

    @handle_errors
    def read_dataframe(
        self, nodes: pd.DataFrame, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.BaseNeuron":
        """Convert a DataFrame into a neuron.

        Parameters
        ----------
        nodes :     pandas.DataFrame
        attrs :     dict or None
                    Arbitrary attributes to include in the neuron.

        Returns
        -------
        core.BaseNeuron
        """
        raise NotImplementedError(
            "Reading DataFrames not implemented for " f"{type(self)}"
        )

    @handle_errors
    def read_buffer(
        self, f: IO, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.BaseNeuron":
        """Read buffer into a single neuron.



        Parameters
        ----------
        f :         IO
                    Readable buffer.
        attrs :     dict | None
                    Arbitrary attributes to include in the neuron.

        Returns
        -------
        core.NeuronObject
        """
        raise NotImplementedError(
            "Reading from buffer not implemented for " f"{type(self)}"
        )

    def read_any_single(
        self, obj, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.BaseNeuron":
        """Attempt to convert an arbitrary object into a neuron.

        Parameters
        ----------
        obj :       typing.IO | pandas.DataFrame | str | os.PathLike
                    Readable buffer, dataframe, path or URL to single file,
                    or parsable string.
        attrs :     dict or None
                    Arbitrary attributes to include in the neuron.

        Returns
        -------
        core.BaseNeuron
        """
        if hasattr(obj, "read"):
            return self.read_buffer(obj, attrs=attrs)
        if isinstance(obj, pd.DataFrame):
            return self.read_dataframe(obj, attrs=attrs)
        if isinstance(obj, os.PathLike):
            if str(obj).endswith(".zip"):
                return self.read_zip(obj, attrs=attrs)
            elif ".tar" in str(obj):
                return self.read_tar(obj, attrs=attrs)
            return self.read_file_path(obj, attrs=attrs)
        if isinstance(obj, str):
            # See if this might be a file (make sure to expand user)
            if os.path.isfile(os.path.expanduser(obj)):
                p = Path(obj).expanduser()
                if p.suffix == ".zip":
                    return self.read_zip(p, attrs=attrs)
                elif p.suffix in (".tar", "tar.gz", "tar.bz"):
                    return self.read_tar(p, attrs=attrs)
                return self.read_file_path(p, attrs=attrs)
            if obj.startswith("http://") or obj.startswith("https://"):
                return self.read_url(obj, attrs=attrs)
            if obj.startswith("ftp://"):
                return self.read_ftp(obj, attrs=attrs)
            return self.read_string(obj, attrs=attrs)
        if isinstance(obj, bytes):
            return self.read_bytes(obj, attrs=attrs)
        raise ValueError(f"Could not read neuron from object of type '{type(obj)}'")

    def read_any_multi(
        self,
        objs,
        include_subdirs=DEFAULT_INCLUDE_SUBDIRS,
        parallel="auto",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "core.NeuronList":
        """Attempt to convert an arbitrary object into a NeuronList,
        potentially in parallel.

        Parameters
        ----------
        obj :               sequence
                            Sequence of anything readable by read_any_single or
                            directory path(s).
        include_subdirs :   bool
                            Whether to include subdirectories of a given
                            directory.
        parallel :          str | bool | int | None
                            "auto" or True for n_cores // 2, otherwise int for
                            number of jobs, or False for serial.
        attrs :             dict or None
                            Arbitrary attributes to include in each TreeNeuron
                            of the NeuronList.

        Returns
        -------
        core.NeuronList

        """
        if not utils.is_iterable(objs):
            objs = [objs]

        if not objs:
            logger.warning("No files found, returning empty NeuronList")
            return self.format_output(objs)

        new_objs = []
        for obj in objs:
            try:
                if is_dir(obj):
                    new_objs.extend(self.files_in_dir(obj, include_subdirs))
                    continue
            except TypeError:
                pass
            new_objs.append(obj)

        # `parallel` can be ("auto", threshold) in which case `threshold`
        # determines at what length we use parallel processing
        if isinstance(parallel, tuple):
            parallel, threshold = parallel
        else:
            threshold = 200

        if (
            isinstance(parallel, str)
            and parallel.lower() == "auto"
            and len(new_objs) < threshold
        ):
            parallel = False

        read_fn = partial(self.read_any_single, attrs=attrs)
        neurons = parallel_read(read_fn, new_objs, parallel)
        return self.format_output(neurons)

    def read_any(
        self,
        obj,
        include_subdirs=DEFAULT_INCLUDE_SUBDIRS,
        parallel="auto",
        limit=None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> "core.NeuronObject":
        """Attempt to read an arbitrary object into a neuron.

        Parameters
        ----------
        obj :       typing.IO | str | os.PathLike | pandas.DataFrame
                    Buffer, path to file or directory, URL, string, or
                    dataframe.

        Returns
        -------
        core.NeuronObject
        """
        if utils.is_iterable(obj) and not hasattr(obj, "read"):
            return self.read_any_multi(obj, parallel, include_subdirs, attrs=attrs)
        else:
            try:
                if is_dir(obj):
                    return self.read_directory(
                        obj, include_subdirs, parallel, limit, attrs=attrs
                    )
            except TypeError:
                pass
            try:
                if os.path.isfile(os.path.expanduser(obj)) and str(obj).endswith(
                    ".zip"
                ):
                    return self.read_zip(obj, parallel, limit, attrs=attrs)
                if os.path.isfile(os.path.expanduser(obj)) and ".tar" in str(obj):
                    return self.read_tar(obj, parallel, limit, attrs=attrs)
                if isinstance(obj, str) and obj.startswith("ftp://"):
                    return self.read_ftp(obj, parallel, limit, attrs=attrs)
            except TypeError:
                pass
            return self.read_any_single(obj, attrs=attrs)

    def parse_filename(self, filename: str) -> dict:
        """Extract properties from filename according to specified formatter.

        Parameters
        ----------
        filename : str

        Returns
        -------
        props :     dict
                    Properties extracted from filename.

        """
        # Make sure we are working with the filename not the whole path
        filename = Path(filename).name

        # Escape all special characters
        fmt = re.escape(self.fmt)

        # Unescape { and }
        fmt = fmt.replace("\\{", "{").replace("\\}", "}")

        # Replace all e.g. {name} with {.*}
        prop_names = []
        for prop in re.findall("{.*?}", fmt):
            prop_names.append(prop[1:-1].replace(" ", ""))
            fmt = fmt.replace(prop, "(.*)")

        # Match
        match = re.search(fmt, filename)

        if not match:
            raise ValueError(f'Unable to match "{self.fmt}" to filename "{filename}"')

        props = {"file": filename}
        for i, prop in enumerate(prop_names):
            for p in prop.split(","):
                # Ignore empty ("{}")
                if not p:
                    continue

                # If datatype was specified
                if ":" in p:
                    p, dt = p.split(":")

                    props[p] = match.group(i + 1)

                    if dt == "int":
                        props[p] = int(props[p])
                    elif dt == "float":
                        props[p] = float(props[p])
                    elif dt == "bool":
                        props[p] = bool(props[p])
                    elif dt == "str":
                        props[p] = str(props[p])
                    else:
                        raise ValueError(
                            f'Unable to interpret datatype "{dt}" ' f"for property {p}"
                        )
                else:
                    props[p] = match.group(i + 1)
        return props

    def _extract_connectors(self, nodes: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Infer outgoing/incoming connectors from data.

        Parameters
        ----------
        nodes :     pd.DataFrame

        Returns
        -------
        Optional[pd.DataFrame]
                    With columns `["node_id", "x", "y", "z", "connector_id", "type"]`
        """
        return


class ImageReader(BaseReader):
    """Reader for image data.


    """

    def __init__(self, output, thin, threshold, dotprop_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.output = output
        self.thin = thin
        self.threshold = threshold
        self.dotprop_kwargs = dotprop_kwargs

    def convert_image(self, data, attrs, header, voxdim, units, space_units):
        """Convert image data to desired output.

        Parameters
        ----------
        data :          np.ndarray
                        Image/Voxel data.
        attrs :         dict
                        Additional attributes to associate with the neuron.
        header :        dict
                        Header information.
        voxdim :        list of numbers
                        Voxel dimensions.
        units :         str | list, optional
                        Units for the neuron (e.g. "1um" or ["4um", "4um", "40um"]).
        space_units :   str, optional
                        Space units (e.g. "um").

        """
        if self.output == "dotprops":
            # If we're trying to get voxels from an image
            if data.ndim == 3:
                if self.threshold:
                    if self.threshold >= 1:
                        data = data >= self.threshold
                    elif self.threshold < 1 and self.threshold > 0:
                        data = data >= (self.threshold * data.max())
                    else:
                        raise ValueError(
                            "Threshold must be either >=1 or 0-1, got "
                            f"{self.threshold}"
                        )

                if self.thin:
                    from skimage.morphology import skeletonize

                    data = skeletonize(data)

                # Convert data to x/y/z coordinates
                # Note we need to multiply units before creating the Dotprops
                # - otherwise the KNN will be wrong
                x, y, z = np.where(data)
                points = np.vstack((x, y, z)).T
                points = points * voxdim

                if not len(points):
                    raise ValueError(
                        f"No points extracted from {self.name_fallback} file. Try lowering the threshold?"
                    )

                x = core.make_dotprops(points, **self.dotprop_kwargs)
            elif data.ndim == 2:
                if data.shape[1] == 3:
                    points, vect, alpha = data, None, None
                elif data.shape[1] == 6:
                    points, vect, alpha = data[:, :3], data[:, 3:6], None
                elif data.shape[1] == 7:
                    points, vect, alpha = data[:, :3], data[:, 3:6], data[:, 6]
                else:
                    raise ValueError(
                        "Expected data to be either (N, 3), (N, 6) "
                        f"or (N, 7) but {self.name_fallback} file contains {data.shape}"
                    )
                # Get `k` either from provided kwargs or the file's header
                k = self.dotprop_kwargs.pop("k", header.get("k", 20))

                x = core.Dotprops(
                    points, k=k, vect=vect, alpha=alpha, **self.dotprop_kwargs
                )
            else:
                raise ValueError(
                    "Data must be 2- or 3-dimensional to extract "
                    f"Dotprops, got {data.ndim}"
                )

            # Set units from space_units (points are already in physical space)
            if space_units:
                if isinstance(space_units, str):
                    x.units = f"1 {space_units}"
                elif len(space_units) == 3:
                    x.units = [f"1 {s}" for s in space_units]
        else:
            if data.ndim == 2:
                logger.warning(
                    f"Data in {self.name_fallback} file is of shape {data.shape} - "
                    "i.e. 2D. Could this be a point cloud or dotprops "
                    "instead of voxels?"
                )
            x = core.VoxelNeuron(data, units=units)

        # Header is special - we do not want to register it
        setattr(x, f"{self.name_fallback.lower()}_header", header)

        # Try adding properties one-by-one. If one fails, we'll keep track of it
        # in the `.meta` attribute
        meta = {}
        for k, v in attrs.items():
            try:
                x._register_attr(k, v)
            except (AttributeError, ValueError, TypeError):
                meta[k] = v

        if meta:
            x.meta = meta

        return x


def parallel_read(read_fn, objs, parallel="auto") -> List["core.NeuronList"]:
    """Read neurons from some objects with the given reader function,
    potentially in parallel.

    Reader function must be picklable.

    Parameters
    ----------
    read_fn :       Callable
    objs :          Iterable
    parallel :      str | bool | int
                    "auto" or True for `n_cores` // 2, otherwise int for number
                    of jobs, or false for serial.

    Returns
    -------
    core.NeuronList

    """
    try:
        length = len(objs)
    except TypeError:
        length = None

    prog = partial(
        config.tqdm,
        desc="Importing",
        total=length,
        disable=config.pbar_hide,
        leave=config.pbar_leave,
    )

    # `parallel` can be ("auto", threshold) in which case `threshold`
    # determines at what length we use parallel processing
    if isinstance(parallel, tuple):
        parallel, threshold = parallel
    else:
        threshold = 200

    if (
        isinstance(parallel, str)
        and parallel.lower() == "auto"
        and not isinstance(length, type(None))
        and length < threshold
    ):
        parallel = False

    if parallel:
        # Do not swap this as `isinstance(True, int)` returns `True`
        if isinstance(parallel, (bool, str)):
            n_cores = max(1, os.cpu_count() // 2)
        else:
            n_cores = int(parallel)

        with mp.Pool(processes=n_cores) as pool:
            results = pool.imap(read_fn, objs)
            neurons = list(prog(results))
    else:
        neurons = [read_fn(obj) for obj in prog(objs)]

    return neurons


def parallel_read_archive(
    read_fn, fpath, file_ext, limit=None, parallel="auto", ignore_hidden=True
) -> List["core.NeuronList"]:
    """Read neurons from a archive (zip or tar), potentially in parallel.

    Reader function must be picklable.

    Parameters
    ----------
    read_fn :       Callable
    fpath :         str | Path
    file_ext :      str | callable
                    File extension to search for - e.g. ".swc". `None` or `''`
                    are interpreted as looking for filenames without extension.
                    To include all files use `'*'`. Can also be callable that
                    accepts a filename and returns True or False depending on
                    if it should be included.
    limit :         int, optional
                    Limit the number of files read from this directory.
    parallel :      str | bool | int
                    "auto" or True for n_cores // 2, otherwise int for number of
                    jobs, or false for serial.
    ignore_hidden : bool
                    Archives zipped on OSX can end up containing a
                    `__MACOSX` folder with files that mirror the name of other
                    files. For example if there is a `123456.swc` in the archive
                    you might also find a `__MACOSX/._123456.swc`. Reading the
                    latter will result in an error. If ignore_hidden=True
                    we will simply ignore all file that starts with "._".

    Returns
    -------
    core.NeuronList

    """
    # Check zip content
    p = Path(fpath)
    to_read = []

    if p.name.endswith(".zip"):
        with ZipFile(p, "r") as zip:
            for i, file in enumerate(zip.filelist):
                fname = file.filename.split("/")[-1]
                if ignore_hidden and fname.startswith("._"):
                    continue
                if callable(file_ext):
                    if file_ext(file):
                        to_read.append(file)
                elif file_ext == "*":
                    to_read.append(file)
                elif file_ext and fname.endswith(file_ext):
                    to_read.append(file)
                elif "." not in file.filename:
                    to_read.append(file)

                if isinstance(limit, int) and i >= limit:
                    break
    elif ".tar" in p.name:  # can be ".tar", "tar.gz" or "tar.bz"
        with tarfile.open(p, "r") as tf:
            for i, file in enumerate(tf):
                fname = file.name.split("/")[-1]
                if ignore_hidden and fname.startswith("._"):
                    continue
                if callable(file_ext):
                    if file_ext(file):
                        to_read.append(file)
                elif file_ext == "*":
                    to_read.append(file)
                elif file_ext and fname.endswith(file_ext):
                    to_read.append(file)
                elif "." not in file.filename:
                    to_read.append(file)

                if isinstance(limit, int) and i >= limit:
                    break

    if isinstance(limit, list):
        to_read = [f for f in to_read if f in limit]
    elif isinstance(limit, slice):
        to_read = to_read[limit]
    elif isinstance(limit, str):
        # Check if limit is a regex
        if rgx.search(limit):
            to_read = [f for f in to_read if re.search(limit, f.filename)]
        else:
            to_read = [f for f in to_read if limit in f.filename]

    prog = partial(
        config.tqdm,
        desc="Importing",
        total=len(to_read),
        disable=config.pbar_hide,
        leave=config.pbar_leave,
    )

    # `parallel` can be ("auto", threshold) in which case `threshold`
    # determines at what length we use parallel processing
    if isinstance(parallel, tuple):
        parallel, threshold = parallel
    else:
        threshold = 200

    if (
        isinstance(parallel, str)
        and parallel.lower() == "auto"
        and len(to_read) < threshold
    ):
        parallel = False

    if parallel:
        # Do not swap this as `isinstance(True, int)` returns `True`
        if isinstance(parallel, (bool, str)):
            n_cores = max(1, os.cpu_count() // 2)
        else:
            n_cores = int(parallel)

        with mp.Pool(processes=n_cores) as pool:
            results = pool.imap(read_fn, to_read)
            neurons = list(prog(results))
    else:
        neurons = [read_fn(obj) for obj in prog(to_read)]

    return neurons


def parallel_read_ftp(
    read_fn,
    server,
    port,
    path,
    file_ext,
    limit=None,
    parallel="auto",
) -> List["core.NeuronList"]:
    """Read neurons from an FTP server, potentially in parallel.

    Reader function must be picklable.

    Parameters
    ----------
    read_fn :       Callable
    server :        str
                    FTP server address.
    port :          int
                    FTP server port.
    path :          str
                    Path to directory containing files or single file.
    file_ext :      str | callable
                    File extension to search for - e.g. ".swc". `None` or `''`
                    are interpreted as looking for filenames without extension.
                    To include all files use `'*'`. Can also be callable that
                    accepts a filename and returns True or False depending on
                    if it should be included.
    limit :         int, optional
                    Limit the number of files read from this directory.
    parallel :      str | bool | int
                    "auto" or True for n_cores // 2, otherwise int for number of
                    jobs, or false for serial.

    Returns
    -------
    core.NeuronList

    """
    # Check if this is a single file
    is_single_file = False
    if "*" not in path:
        if isinstance(file_ext, str) and path.endswith(file_ext):
            is_single_file = True
        elif callable(file_ext) and file_ext(path.rsplit("/", 1)[1]):
            is_single_file = True

    if is_single_file:
        path, fname = path.rsplit("/", 1)
        to_read = [fname]
    else:
        pattern = ""
        # Check if path contains a "*." pattern - e.g. something like "*_raw.swc"
        if "*" in path:
            path, fname = path.rsplit("/", 1)
            pattern = fname

        # Remove leading /
        if path.startswith("/"):
            path = path[1:]

        # First check content
        with FTP() as ftp:
            ftp.connect(server, port)  # connect to server
            ftp.login()  # anonymous login
            ftp.cwd(path)  # change to path

            # Read content
            content = []
            ftp.retrlines(f"LIST {pattern}", content.append)

        # Parse content into filenames
        to_read = []
        for line in content:
            if not line:
                continue
            file = line.split()[-1].strip()

            if callable(file_ext):
                if file_ext(file):
                    to_read.append(file)
            elif file_ext == "*":
                to_read.append(file)
            elif file_ext and fname.endswith(file_ext):
                to_read.append(file)

    if isinstance(limit, int):
        to_read = to_read[:limit]
    elif isinstance(limit, list):
        to_read = [f for f in to_read if f in limit]
    elif isinstance(limit, slice):
        to_read = to_read[limit]
    elif isinstance(limit, str):
        # Check if limit is a regex
        if rgx.search(limit):
            to_read = [f for f in to_read if re.search(limit, f)]
        else:
            to_read = [f for f in to_read if limit in f]

    if not to_read:
        return []

    prog = partial(
        config.tqdm,
        desc="Loading",
        total=len(to_read),
        disable=config.pbar_hide,
        leave=config.pbar_leave,
    )

    # `parallel` can be ("auto", threshold) in which case `threshold`
    # determines at what length we use parallel processing
    if isinstance(parallel, tuple):
        parallel, threshold = parallel
    else:
        threshold = 200

    if (
        isinstance(parallel, str)
        and parallel.lower() == "auto"
        and len(to_read) < threshold
    ):
        parallel = False

    if parallel:
        # Do not swap this as `isinstance(True, int)` returns `True`
        if isinstance(parallel, (bool, str)):
            n_cores = max(1, os.cpu_count() // 2)
        else:
            n_cores = int(parallel)

        # We can't send the FTP object to the process (because its socket is not pickleable)
        # Instead, we need to initialize a new FTP connection in each process via a global variable
        with mp.Pool(
            processes=n_cores, initializer=_ftp_pool_init, initargs=(server, port, path)
        ) as pool:
            results = pool.imap(partial(read_fn, ftp="GLOBAL"), to_read)
            neurons = list(prog(results))
    else:
        with FTP() as ftp:
            ftp.connect(server, port)
            ftp.login()
            ftp.cwd(path)

            neurons = [read_fn(file, ftp=ftp) for file in prog(to_read)]

    return neurons


def _ftp_pool_init(server, port, path):
    global _FTP
    _FTP = FTP()
    _FTP.connect(server, port)
    _FTP.login()
    _FTP.cwd(path)


def parse_precision(precision: Optional[int]):
    """Convert bit width into int and float dtypes.

    Parameters
    ----------
    precision : int
        16, 32, 64, or None

    Returns
    -------
    tuple
        Integer numpy dtype, float numpy dtype

    Raises
    ------
    ValueError
        Unknown precision.
    """
    INT_DTYPES = {16: np.int16, 32: np.int32, 64: np.int64, None: None}
    FLOAT_DTYPES = {16: np.float16, 32: np.float32, 64: np.float64, None: None}

    try:
        return (INT_DTYPES[precision], FLOAT_DTYPES[precision])
    except KeyError:
        raise ValueError(
            f"Unknown precision {precision}. Expected on of the following: 16, 32 (default), 64 or None"
        )


class ReadError(Exception):
    """Error raised when reading a file fails."""


def is_dir(path: os.PathLike) -> bool:
    """Check if path is a directory.

    The main purpose of this function is to catch
    *.file_ext at the end of the path.

    Parameters
    ----------
    path : os.PathLike
        Path to check.

    Returns
    -------
    bool
        True if path is a directory.

    """
    # Expand user
    path = os.path.expanduser(path)

    # Remove any trailing *.file_ext
    path = path.split("*")[0]

    return os.path.isdir(path)
