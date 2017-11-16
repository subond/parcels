import subprocess
from os import path, environ, makedirs
from tempfile import gettempdir
from struct import calcsize
try:
    from os import getuid
except:
    # Windows does not have getuid(), so define to simply return 'tmp'
    def getuid():
        return 'tmp'


def get_package_dir():
    return path.abspath(path.join(path.dirname(__file__), path.pardir))


def get_cache_dir():
    directory = path.join(gettempdir(), "parcels-%s" % getuid())
    if not path.exists(directory):
        makedirs(directory)
    return directory


class Compiler(object):
    """A compiler object for creating and loading shared libraries.

    :arg cc: C compiler executable (can be overriden by exporting the
        environment variable ``CC``).
    :arg ld: Linker executable (optional, if ``None``, we assume the compiler
        can build object files and link in a single invocation, can be
        overridden by exporting the environment variable ``LDSHARED``).
    :arg cppargs: A list of arguments to the C compiler (optional).
    :arg ldargs: A list of arguments to the linker (optional)."""

    def __init__(self, cc, ld=None, cppargs=[], ldargs=[]):
        self._cc = environ.get('CC', cc)
        self._ld = environ.get('LDSHARED', ld)
        self._cppargs = cppargs
        self._ldargs = ldargs

    def compile(self, src, obj, log):
        cc = [self._cc] + self._cppargs + ['-o', obj, src] + self._ldargs
        with open(log, 'w') as logfile:
            logfile.write("Compiling: %s\n" % " ".join(cc))
            try:
                subprocess.check_call(cc, stdout=logfile, stderr=logfile)
            except OSError:
                err = """OSError during compilation
Please check if compiler exists: %s""" % self._cc
                raise RuntimeError(err)
            except subprocess.CalledProcessError:
                err = """Error during compilation:
Compilation command: %s
Source file: %s
Log file: %s""" % (" ".join(cc), src, logfile.name)
                raise RuntimeError(err)


class GNUCompiler(Compiler):
    """A compiler object for the GNU Linux toolchain.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional)."""
    def __init__(self, cppargs=[], ldargs=[]):
        opt_flags = ['-g', '-O3']
        arch_flag = ['-m64' if calcsize("P") is 8 else '-m32']
        cppargs = ['-Wall', '-fPIC', '-I%s' % path.join(get_package_dir(), 'include')] + opt_flags + cppargs
        cppargs += arch_flag
        ldargs = ['-shared'] + ldargs + arch_flag
        super(GNUCompiler, self).__init__("gcc", cppargs=cppargs, ldargs=ldargs)
