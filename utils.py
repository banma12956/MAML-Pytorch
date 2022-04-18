# useful functions from vamb/vambtools.py in VAMB
import torch
import random
import numpy as np
import gzip as gzip
import bz2 as bz2
import lzma as lzma
from torch.utils.data.dataset import TensorDataset
from _vambtools import _kmercounts, _overwrite_matrix
import collections
# from hashlib import md5 as _md5


class PushArray:
    """Data structure that allows efficient appending and extending a 1D Numpy array.
    Intended to strike a balance between not resizing too often (which is slow), and
    not allocating too much at a time (which is memory inefficient).

    Usage:
    >>> arr = PushArray(numpy.float64)
    >>> arr.append(5.0)
    >>> arr.extend(numpy.linspace(4, 3, 3))
    >>> arr.take() # return underlying Numpy array
    array([5. , 4. , 3.5, 3. ])
    """

    __slots__ = ['data', 'capacity', 'length']

    def __init__(self, dtype, start_capacity=1<<16):
        self.capacity = start_capacity
        self.data = np.empty(self.capacity, dtype=dtype)
        self.length = 0

    def __len__(self):
        return self.length

    def _setcapacity(self, n):
        self.data.resize(n, refcheck=False)
        self.capacity = n

    def _grow(self, mingrowth):
        """Grow capacity by power of two between 1/8 and 1/4 of current capacity, though at
        least mingrowth"""
        growth = max(int(self.capacity * 0.125), mingrowth)
        nextpow2 = 1 << (growth - 1).bit_length()
        self._setcapacity(self.capacity + nextpow2)

    def append(self, value):
        if self.length == self.capacity:
            self._grow(64)

        self.data[self.length] = value
        self.length += 1

    def extend(self, values):
        lenv = len(values)
        if self.length + lenv > self.capacity:
            self._grow(lenv)

        self.data[self.length:self.length+lenv] = values
        self.length += lenv

    def take(self):
        "Return the underlying array"
        self._setcapacity(self.length)
        return self.data

    def clear(self, force=False):
        "Empties the PushArray. If force is true, also truncates the underlying memory."
        self.length = 0
        if force:
            self._setcapacity(0)

def zscore(un, an, axis=None, inplace=False):
    """Calculates zscore for an array. A cheap copy of scipy.stats.zscore.

    Inputs:
        array: Numpy array to be normalized
        axis: Axis to operate across [None = entrie array]
        inplace: Do not create new array, change input array [False]

    Output:
        If inplace is True: None
        else: New normalized Numpy-array"""

    if axis is not None and axis >= an.ndim:
        raise np.AxisError('array only has {} axes'.format(array.ndim))

    if inplace and not np.issubdtype(an.dtype, np.floating):
        raise TypeError('Cannot convert a non-float array to zscores')

    array = np.vstack((un, an))
    mean = array.mean(axis=axis)
    std = array.std(axis=axis)
    '''
    print("mean", mean.shape)
    print("std", std.shape)
    print("array", array.shape)
    print("un", un.shape)
    print("an", an.shape)
    exit()

    if axis is None:
        if std == 0:
            std = 1 # prevent divide by zero

    else:
        std[std == 0.0] = 1 # prevent divide by zero
        shape = tuple(dim if ax != axis else 1 for ax, dim in enumerate(array.shape))
        mean.shape, std.shape = shape, shape
    '''
    std[std == 0.0] = 1
    if inplace:
        un -= mean
        an -= mean
        un /= std
        an /= std
        return None
    else:
        return (un - mean) / std, (an - mean) / std

def torch_inplace_maskarray(array, mask):
    """In-place masking of a Tensor, i.e. if `mask` is a boolean mask of same
    length as `array`, then array[mask] == torch_inplace_maskarray(array, mask),
    but does not allocate a new tensor.
    """

    if len(mask) != len(array):
        raise ValueError('Lengths of array and mask must match')
    elif array.dim() != 2:
        raise ValueError('Can only take a 2 dimensional-array.')

    np_array = array.numpy()
    np_mask = np.frombuffer(mask.numpy(), dtype=np.uint8)
    index = _overwrite_matrix(np_array, np_mask)
    array.resize_((index, array.shape[1]))
    return array

class DataSampler:
    def __init__(self, data, batch_size=256, cuda=False):
        self.data = list(map(lambda x: torch.from_numpy(x), data))
        if cuda:
            self.data = list(map(lambda x: x.cuda(), self.data))
        self.size = len(data[0])
        self.batch_size = batch_size

    def sample(self, size=None):
        if size == None:
            size = self.batch_size
        indices = random.sample(range(0, self.size), size)
        return list(map(lambda x: x[indices], self.data))

    def __len__(self):
        return self.size

def make_dataloader(data, batch_size=256, cuda=False):  # TODO: remove
    data = list(map(lambda x: torch.from_numpy(x), data))
    # depthstensor = torch.from_numpy(rpkms)
    # tnftensor = torch.from_numpy(tnfs)
    # dataset = TensorDataset(depthstensor, tnftensor)
    dataset = TensorDataset(*data)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, 
                 batch_size=256, num_workers=2)
                 #, pin_memory=cuda)     # TODO: about workers, batch_size

    return dataloader

class Reader:
    """Use this instead of `open` to open files which are either plain text,
    gzipped, bzip2'd or zipped with LZMA.

    Usage:
    >>> with Reader(file, readmode) as file: # by default textmode
    >>>     print(next(file))
    TEST LINE
    """

    def __init__(self, filename, readmode='r'):
        if readmode not in ('r', 'rt', 'rb'):
            raise ValueError("the Reader cannot write, set mode to 'r' or 'rb'")
        if readmode == 'r':
            self.readmode = 'rt'
        else:
            self.readmode = readmode

        self.filename = filename

        with open(self.filename, 'rb') as f:
            signature = f.peek(8)[:8]

        # Gzipped files begin with the two bytes 0x1F8B
        if tuple(signature[:2]) == (0x1F, 0x8B):
            self.filehandle = gzip.open(self.filename, self.readmode)

        # bzip2 files begin with the signature BZ
        elif signature[:2] == b'BZ':
            self.filehandle = bz2.open(self.filename, self.readmode)

        # .XZ files begins with 0xFD377A585A0000
        elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
            self.filehandle = lzma.open(self.filename, self.readmode)

        # Else we assume it's a text file.
        else:
            self.filehandle = open(self.filename, self.readmode)

    def close(self):
        self.filehandle.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        return self.filehandle

class FastaEntry:
    """One single FASTA entry. Instantiate with string header and bytearray
    sequence."""

    basemask = bytearray.maketrans(b'acgtuUswkmyrbdhvnSWKMYRBDHV',
                                   b'ACGTTTNNNNNNNNNNNNNNNNNNNNN')
    __slots__ = ['header', 'sequence']

    def __init__(self, header, sequence):
        if len(header) > 0 and (header[0] in ('>', '#') or header[0].isspace()):
            raise ValueError('Header cannot begin with #, > or whitespace')
        if '\t' in header:
            raise ValueError('Header cannot contain a tab')

        masked = sequence.translate(self.basemask, b' \t\n\r')
        stripped = masked.translate(None, b'ACGTN')
        if len(stripped) > 0:
            bad_character = chr(stripped[0])
            msg = "Non-IUPAC DNA byte in sequence {}: '{}'"
            raise ValueError(msg.format(header, bad_character))

        self.header = header
        self.sequence = masked

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return '>{}\n{}'.format(self.header, self.sequence.decode())

    def format(self, width=60):
        sixtymers = range(0, len(self.sequence), width)
        spacedseq = '\n'.join([self.sequence[i: i+width].decode() for i in sixtymers])
        return '>{}\n{}'.format(self.header, spacedseq)

    def __getitem__(self, index):
        return self.sequence[index]

    def __repr__(self):
        return '<FastaEntry {}>'.format(self.header)

    def kmercounts(self, k):
        if k < 1 or k > 10:
            raise ValueError('k must be between 1 and 10 inclusive')

        counts = np.zeros(1 << (2*k), dtype=np.int32)
        _kmercounts(self.sequence, k, counts)
        return counts

def byte_iterfasta(filehandle, comment=b'#'):
    """Yields FastaEntries from a binary opened fasta file.

    Usage:
    >>> with Reader('/dir/fasta.fna', 'rb') as filehandle:
    ...     entries = byte_iterfasta(filehandle) # a generator

    Inputs:
        filehandle: Any iterator of binary lines of a FASTA file
        comment: Ignore lines beginning with any whitespace + comment

    Output: Generator of FastaEntry-objects from file
    """

    # Make it work for persistent iterators, e.g. lists
    line_iterator = iter(filehandle)
    # Skip to first header
    try:
        for probeline in line_iterator:
            stripped = probeline.lstrip()
            if stripped.startswith(comment):
                pass

            elif probeline[0:1] == b'>':
                break

            else:
                raise ValueError('First non-comment line is not a Fasta header')

        else: # no break
            raise ValueError('Empty or outcommented file')

    except TypeError:
        errormsg = 'First line does not contain bytes. Are you reading file in binary mode?'
        raise TypeError(errormsg) from None

    header = probeline[1:-1].decode()
    buffer = list()

    # Iterate over lines
    for line in line_iterator:
        if line.startswith(comment):
            pass

        elif line.startswith(b'>'):
            yield FastaEntry(header, bytearray().join(buffer))
            buffer.clear()
            header = line[1:-1].decode()

        else:
            buffer.append(line)

    yield FastaEntry(header, bytearray().join(buffer))

def read_clusters(filehandle, min_size=1):
    """Read clusters from a file as created by function `writeclusters`.

    Inputs:
        filehandle: An open filehandle that can be read from
        min_size: Minimum number of contigs in cluster to be kept

    Output: A {clustername: set(contigs)} dict"""

    contigsof = collections.defaultdict(set)

    for line in filehandle:
        stripped = line.strip()

        if not stripped or stripped[0] == '#':
            continue

        clustername, contigname = stripped.split('\t')
        contigsof[clustername].add(contigname)

    contigsof = {cl: co for cl, co in contigsof.items() if len(co) >= min_size}

    return contigsof

def write_clusters(filehandle, clusters, max_clusters=None,
                 header=None, rename=True):
    """Writes clusters to an open filehandle.
    Inputs:
        filehandle: An open filehandle that can be written to
        clusters: An iterator generated by function `clusters` or a dict
        max_clusters: Stop printing after this many clusters [None]
        min_size: Don't output clusters smaller than N contigs
        header: Commented one-line header to add
        rename: Rename clusters to "cluster_1", "cluster_2" etc.

    Outputs:
        clusternumber: Number of clusters written
        ncontigs: Number of contigs written
    """

    if not hasattr(filehandle, 'writable') or not filehandle.writable():
        raise ValueError('Filehandle must be a writable file')

    # Special case to allows dicts even though they are not iterators of
    # clustername, {cluster}
    #if isinstance(clusters, dict):
    #    clusters = clusters.items()

    if max_clusters is not None and max_clusters < 1:
        raise ValueError('max_clusters must None or at least 1, not {}'.format(max_clusters))

    if header is not None and len(header) > 0:
        if '\n' in header:
            raise ValueError('Header cannot contain newline')

        if header[0] != '#':
            header = '# ' + header

        print(header, file=filehandle)

    clusternumber = 0
    ncontigs = 0

    for clustername, contigs in clusters.items():
        # if len(contigs) < min_size:
        #     continue

        if rename:
            clustername = 'cluster_' + str(clusternumber + 1)

        for contig in contigs:
            print(clustername, contig, sep='\t', file=filehandle)
        filehandle.flush()

        clusternumber += 1
        ncontigs += len(contigs)

        if clusternumber == max_clusters:
            break

    return clusternumber, ncontigs

def validate_input_array(array):
    "Returns array similar to input array but C-contiguous and with own data."
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    if not array.flags['OWNDATA']:
        array = array.copy()

    assert (array.flags['C_CONTIGUOUS'] and array.flags['OWNDATA'])
    return array

def read_npz(file_p):
    """Loads array in .npz-format

    Input: Open file or path to file with npz-formatted array

    Output: A Numpy array
    """

    npz = np.load(file_p)
    array = validate_input_array(npz['arr_0'])
    npz.close()

    return array

def write_npz(file_p, array):
    """Writes a Numpy array to an open file or path in .npz format

    Inputs:
        file: Open file or path to file
        array: Numpy array

    Output: None
    """
    np.savez_compressed(file_p, array)

def _split_bin(binname, headers, separator, bysample=collections.defaultdict(set)):
    "Split a single bin by the prefix of the headers"

    bysample.clear()
    for header in headers:
        if not isinstance(header, str):
            raise TypeError('Can only split named sequences, not of type {}'.format(type(header)))

        sample, _sep, identifier = header.partition(separator)

        if not identifier:
            raise KeyError("Separator '{}' not in sequence label: '{}'".format(separator, header))

        bysample[sample].add(header)

    for sample, splitheaders in bysample.items():
        newbinname = "{}{}{}".format(sample, separator, binname)
        yield newbinname, splitheaders

def _binsplit_generator(cluster_iterator, separator):
    "Return a generator over split bins with the function above."
    for binname, headers in cluster_iterator:
        for newbinname, splitheaders in _split_bin(binname, headers, separator):
            yield newbinname, splitheaders

def binsplit(clusters, separator):
    """Splits a set of clusters by the prefix of their names.
    The separator is a string which separated prefix from postfix of contignames. The
    resulting split clusters have the prefix and separator prepended to them.

    clusters can be an iterator, in which case this function returns an iterator, or a dict
    with contignames: set_of_contignames pair, in which case a dict is returned.

    Example:
    >>> clusters = {"bin1": {"s1-c1", "s1-c5", "s2-c1", "s2-c3", "s5-c8"}}
    >>> binsplit(clusters, "-")
    {'s2-bin1': {'s1-c1', 's1-c3'}, 's1-bin1': {'s1-c1', 's1-c5'}, 's5-bin1': {'s1-c8'}}
    """
    if iter(clusters) is clusters: # clusters is an iterator
        return _binsplit_generator(clusters, separator)

    elif isinstance(clusters, dict):
        return dict(_binsplit_generator(clusters.items(), separator))

    else:
        raise TypeError("clusters must be iterator of pairs or dict")
