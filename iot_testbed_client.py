#!/usr/bin/env python3

import sys
import json
import copy
import re
from datetime import datetime, timedelta, date, time
from pathlib import Path
import warnings
import itertools
import pprint
import traceback

from requests import Session
from requests.compat import urljoin
import shutil

import dateutil.parser

import unittest

import dataclasses
import typing

default_config = {
    "server": "https://research.iottestbed.disi.unitn.it",
    "cacert": Path(__file__).resolve().parent.joinpath(
        "testbediot-disi-unitn-it-chain.pem")
}

import importlib.util

ISLAND_LIST = ['DEPT', 'HALL-A', 'all']

@dataclasses.dataclass
class Config:
    CONFIG_FILE_NAME = '.iottestbed.config.json'

    server: str # Maybe it should be uri
    cacert: str # Maybe it should be path
    token: typing.Optional[str] = None

    calendar_island: str = 'all'

    hooks: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    dest_dir: Path = Path('./')

    @staticmethod
    def load(args: dict[str, typing.Any], cwd_path: Path, home_path: Path, default_config: typing.Any) -> 'tuple[bool,Config]':

        def get_config(configFile):
            with configFile.open() as fh:
                config = json.load(fh)

            return config

        # Get all parent paths from the current working directory (including this directory)
        parents = [p for p in cwd_path.parents]
        parents = [cwd_path, *parents]

        # Reverse the order of the directories 
        # (we want parent folder first so that we can then overwrite their 
        #  configurations with the configuration in the subdirectories)
        parents = parents[::-1]

        # Make sure there is at least the home folder (as lowest priority)
        parents = [home_path, *parents]

        # Find the paths in which there is an actual configuration
        parents = [p.joinpath(Config.CONFIG_FILE_NAME) for p in parents]
        parents = [p for p in parents if p.is_file()]

        try:
            # Get all the configurations
            parents = [get_config(p) for p in parents]
        except json.JSONDecodeError as ex:
            # If there is an error when loading the configuration, fallback 
            # to the default configuration.
            # In this way we are sure that we always have a working --help
            print('Could not decode all json configurations. Falling back to default configuration')
            print(ex)
            print('') 
            # TODO: This error message is pretty bad
            return False, Config(**default_config)

        # With as base defaults default_config, overwrite the config, giving highest 
        # priority to the configurations to the directories closest to the cwd
        config = {**default_config}
        for p in parents:
            config = {**config, **p}

        # Use args if available
        for k in ['server', 'cacert', 'token']:
            if k in args and args[k] is not None:
                config[k] = args[k]

        return True, Config(**config)

def add_hooks(func, hooks, hooks_data):
    def internal(*args, **kwargs):
        data = copy.deepcopy(hooks_data)
        data['inputs'] = (args, kwargs)

        data['output'] = func(*args, **kwargs)

        for hook in hooks:
            hook = str(Path(hook).resolve())
            try:
                try:
                    from importlib.machinery import SourceFileLoader
                    # importing the add module from the calculator package using the path
                    module_handle = SourceFileLoader("hook_module", hook).load_module()
                except NameError:
                    print('Hook module not found')
                else:
                    module_handle.hook(**data)

            except Exception as e:
                print(f'Exception inside a hook module ({hook})')
                print(e)
            except:
                print('Unexpected error:')
                traceback.print_exc()

        return data['output']

    return internal

class Formatter:
    @staticmethod
    def is_humanize_enabled():
        return (importlib.util.find_spec('humanize') is not None)

    @staticmethod
    def is_color_enabled():
        # Check that the colored library is available and check that stdout is a tty.
        # If not do not enable colors
        return (importlib.util.find_spec('colored') is not None) and sys.stdout.isatty()

    @staticmethod
    def format_time_span(begin: str|datetime, end: str|datetime):
        HAS_HUMANIZE = Formatter.is_humanize_enabled()

        if not isinstance(begin, datetime):
            begin = dateutil.parser.isoparse(begin)

        if not isinstance(end, datetime):
            end = dateutil.parser.isoparse(end)

        formattedBeg = begin.isoformat()
        formattedEnd = end.isoformat()
        formattedDelta = None

        if HAS_HUMANIZE:
            import humanize
            formattedBeg = humanize.naturaltime(begin)
            formattedEnd = humanize.naturaltime(end)
            formattedDelta = humanize.precisedelta(end - begin, minimum_unit='minutes', format='%.0f')

        if (formattedBeg == formattedEnd) and (HAS_HUMANIZE):
            return {
                'begin': formattedBeg, 
                'end': formattedEnd,
                'delta': formattedDelta,
            }
        else:
            return {
                'begin': formattedBeg,
                'end': formattedEnd,
            }

    @staticmethod
    def format_size(nBytes: int):
        HAS_HUMANIZE = Formatter.is_humanize_enabled()

        if not HAS_HUMANIZE:
            return f'{nBytes} bytes'

        import humanize

        formatted_bytes = humanize.naturalsize(nBytes, False, False, '%.3f')
        return f'{formatted_bytes} ({nBytes} bytes)'

    @staticmethod
    def format_color_calendar_item(busy):
        HAS_COLORED = Formatter.is_color_enabled()

        if not HAS_COLORED:
            return ''

        from colored import Fore, Style

        STYLES = {
            'is_owner': Style.underline,
            'completed': Fore.red,
            'not_owner_or_downloaded': Style.dim
        }

        begColor = []
        begColor.append(Style.reset)

        if dateutil.parser.isoparse(busy['end']) < datetime.now().astimezone():
            if not (busy['is_owner'] and 'Completed' in busy['desc']):
                begColor.append(STYLES['not_owner_or_downloaded'])

        if busy['is_owner']:
            begColor.append(STYLES['is_owner'])

            if 'Completed' in busy['desc']:
                begColor.append(STYLES['completed'])

        return ''.join(begColor)

    @staticmethod
    def format_calendar(busyList, show_island: bool = True):
        if len(busyList) == 0:
            print("No reservation or job scheduled")
            return 

        def format_calendar_item(busy):
            # First make a co
            res = { }

            res['time_span'] = Formatter.format_time_span(busy['begin'], busy['end'])
            if busy['is_owner']:
                res['desc'] = busy['desc'].replace('\n', ' ') + ' '
            else:
                res['desc'] = ''

            res['color'] = Formatter.format_color_calendar_item(busy)
            res['island_name'] = busy['island']['name']
            res['begin'] = busy['begin']
            res['end'] = busy['end']

            return res

        def print_fomatted_calendar_item(busy, show_island: bool):
            print(f"{busy['color']}" # Color section
                  f"{busy['desc'].center(minO1)}" # Description
                  f"From " # Start of time span section
                  f"{busy['time_span']['begin'].center(minF1)} " # Put the starting time
                  f"{'for ' if 'delta' in busy['time_span'] else 'to  '}" # Put "for" if a time delta follows, otherwise "to"
                  f"{busy['time_span'].get('delta', busy['time_span']['end']).center(minF2)}" # Put the time delta if we have it otherwise use the end time
                  , end="")

            if show_island:
                print(
                  f" on island "
                  f"{busy['island_name'].center(minI1)}"
                  , end="")
                  
            print(f" ({busy['begin']} to {busy['end']})", end="")

            print("")

        busyList = list(map(format_calendar_item, busyList))

        minF1 = max(len(busy['time_span']['begin']) for busy in busyList)
        minF2 = max(len(busy['time_span'].get('delta', busy['time_span']['end'])) for busy in busyList)

        minO1 = max(len(busy['desc']) for busy in busyList)

        minI1 = max(len(busy['island_name']) for busy in busyList)

        print("Testbed is busy:")
        for busy in busyList:
            print_fomatted_calendar_item(busy, show_island=show_island)

class ApiError(ValueError):
    def __init__(self, contextMessage, code, message):
        super().__init__(self, f"{contextMessage}: {code} {message}")

        self.contextMessage = contextMessage
        self.code = code
        self.message = message

    def __str__(self):
        return f"{self.contextMessage}: " \
            f"{','.join(itertools.chain.from_iterable(self.message.values()))}"


class Job:
    """
    Class to represent a single job on testbed, the purpose of this class is
    help set and modify in the correct way each parameter
    """
    def __init__(self, source=None, cwd=Path()):
        self.duration = None
        self.start_time = None
        self.binaries = []

        if source is None:
            return

        if not isinstance(source, dict):
            raise ValueError("source must be a dictionary")

        source = copy.deepcopy(source)
        self._cwd = cwd

        # correction from legacy format
        # first ts_init
        if 'start_time' not in source and 'ts_init' in source:
            warnings.warn("Deprecated use of ts_init, change it in start_time",
                          FutureWarning)
            self.start_time = source.pop('ts_init')

        if 'binaries' not in source and 'image' in source:
            warnings.warn("Deprecated use of image, use binaries",
                          FutureWarning)
            tmp = source.pop('image')
        else:
            tmp = source.pop('binaries')

        if not isinstance(tmp, list):
            tmp = [tmp, ]

        for b in tmp:
            if isinstance(b, dict):

                if 'bin_file' not in b and 'file' in b:
                    warnings.warn(
                        "Deprecated use of file in binaries, use bin_file",
                        FutureWarning
                    )
                    b['bin_file'] = b.pop('file')

                if 'targets' not in b and 'target' in b:
                    warnings.warn("Deprecated use of target, use targets",
                                  FutureWarning)
                    b['targets'] = b.pop('target')

                self.binaries.append(Job.Binary(**b))
            else:
                raise ValueError(
                    "binaries field should be a dictionary or list of "
                    "dicitionaries"
                )

        if 'orchestrator' in source:
            if not isinstance(source['orchestrator'], dict):
                raise ValueError("orchestrator element should be a dictionary")

        elif 'python_script' in source:
            warnings.warn(
                "Deprecated use of python_script, change it in orchestrator",
                FutureWarning
            )
            source['orchestrator'] = {'type': 'python',
                                      'file': source.pop('python_script'),
                                      'run': 'run_test'
                                      }

        if 'extra_files' in source:
            efiles = source.pop('extra_files')
            if isinstance(efiles, list):
                for e in efiles:
                    if not isinstance(e, str):
                        raise ValueError(
                            f"Wrong format in extra_files, error on {e}"
                        )
                    self.add_extra_file(e)
            elif isinstance(efiles, str):
                self.add_extra_file(efiles)
            else:
                raise ValueError(
                    "Wrong format of extra_files, should be a list of string "
                    "or a string"
                )

        self.__dict__ = {**self.__dict__, **source}

    class Binary:
        """
        Binary class will contain all data about a binary used to all or some
        node in the testbed
        """
        def __init__(self, hardware, bin_file, targets, programAddress=None,
                     **kargs):
            self.hardware = hardware
            self.bin_file = bin_file
            self.targets = targets
            self.programAddress = programAddress

            self.__dict__ = {**self.__dict__, **kargs}

        def validate(self, cwd, filenames):
            errors = []
            warnings = []

            if not isinstance(self.hardware, str):
                errors.append(
                    "Hardware should be a string with platfoem name for this "
                    "binary file"
                )

            if isinstance(self.bin_file, str):
                self.bin_file = Path(self.bin_file)

            if not isinstance(self.bin_file, Path):
                errors.append("file must be a string or a Path object")

            if not cwd.joinpath(self.bin_file).is_file():
                errors.append(f"File {self.bin_file} not found")

            if self.bin_file.name in filenames:
                warnings.append(f"file {self.bin_file} will be renamed")

            if not isinstance(self.targets, list):
                self.targets = [self.targets, ]

            for t in self.targets:
                if not isinstance(t, (int, str)):
                    errors.append(
                        "targets must be a str, a int or a list of them"
                    )

            return errors, warnings

        def toJSON(self):
            tmp = copy.deepcopy(self)
            if isinstance(tmp.bin_file, Path):
                tmp.bin_file = str(tmp.bin_file)
            return json.dumps(tmp.__dict__)

    def _add_file(self, file_ref):
        if isinstance(file_ref, Path):
            return file_ref

        elif isinstance(file_ref, str):
            return Path(file_ref)

        else:
            raise TypeError("file_ref should be Path or str object")

    def add_extra_file(self, file):
        if not hasattr(self, 'extra_files'):
            self.extra_files = [self._add_file(file), ]
        else:
            self.extra_files.append(self._add_file(file))

    def validate(self):
        errors = []
        warnings = []

        filenames = ['testFile.json', ]

        if not isinstance(self.binaries, list):
            self.binaries = [self.binaries, ]

        for b in self.binaries:
            e, w = b.validate(self._cwd, filenames)
            errors += e
            warnings += w

        if hasattr(self, 'orchestrator'):
            if not isinstance(self.orchestrator, dict):
                errors.append("orchestrator should be a dictionary")
            else:
                if 'file' not in self.orchestrator:
                    errors.append("Orchestartor should have a file filed")
                elif not isinstance(self.orchestrator['file'], (str, Path)):
                    errors.append(
                        "File field of orchestrator should be a string or a "
                        "Path object"
                    )
                else:
                    if isinstance(self.orchestrator['file'], str):
                        self.orchestrator['file'] = Path(
                            self.orchestrator['file']
                        )

                    if not self._cwd.joinpath(
                            self.orchestrator['file']).is_file():
                        errors.append(
                            f"orchestrator file {self.orchestrator['file']} "
                            "not found"
                        )

                    elif self.orchestrator['file'].name in filenames:
                        # will not renamed, it is an error
                        errors.append("duplicated filename for orchestrator "
                                      f"file: {self.orchestrator['file'].name}"
                                      )
                    else:
                        filenames.append(self.orchestrator['file'].name)

        if hasattr(self, 'extra_files'):
            if not isinstance(self.extra_files, list):
                self.extra_files = [self.extra_files, ]

            tmp = self.extra_files.copy()
            self.extra_files = []
            for e in tmp:
                if not isinstance(e, (str, Path)):
                    errors.append(
                        "extra_files must be a str, a Path or a list of them")
                else:
                    if isinstance(e, str):
                        e = Path(e)

                    if not self._cwd.joinpath(e).is_file():
                        errors.append(f"extra file {e} not found")
                    elif e.name in filenames:
                        # extra_files will not be ranamed, it is an error
                        errors.append(f"duplicate filename {e} ")
                    else:
                        self.extra_files.append(e)
                        filenames.append(e.name)

        return errors, warnings

    def toJSON(self, indent=4):
        tmp = copy.deepcopy(self)
        delattr(tmp, '_cwd')

        if isinstance(tmp.start_time, datetime):
            tmp.start_time = tmp.start_time.strftime("%Y-%m-%d %H:%M")

        btmp = tmp.binaries
        tmp.binaries = []
        for b in btmp:
            tmp.binaries.append(json.loads(b.toJSON()))

        if hasattr(tmp, 'orchestrator'):
            if isinstance(tmp.orchestrator, dict):
                if ('file' in tmp.orchestrator and
                    not isinstance(tmp.orchestrator['file'], str)):
                    tmp.orchestrator['file'] = str(tmp.orchestrator['file'])

            else:
                raise TypeError(
                    "Dont't know how to handle an orchestrator element that "
                    "is not a dictionary"
                )

        if hasattr(tmp, 'extra_files') and isinstance(tmp.extra_files, list):
            etmp = tmp.extra_files
            tmp.extra_files = []
            for e in etmp:
                if isinstance(e, Path):
                    tmp.extra_files.append(str(e))
                elif isinstance(e, str):
                    tmp.extra_files.append(e)
                else:
                    raise TypeError(
                        "Don't know hot to handle extra_file element is not "
                        "a Path or str"
                    )

        return json.dumps(tmp.__dict__, indent=indent)

    def attachedFiles(self):
        """ return a list of file used by this job """
        def _cwd_path(f):
            return self._cwd.joinpath(f)

        # binaries files
        for b in self.binaries:
            yield _cwd_path(b.bin_file)

        # orchestartor
        if hasattr(self, 'orchestrator'):
            yield _cwd_path(self.orchestrator['file'])

        # extra file
        if hasattr(self, 'extra_files'):
            if isinstance(self.extra_files, list):
                for f in self.extra_files:
                    yield _cwd_path(f)
            else:
                yield Path(self.extra_files)


class TestJobs(unittest.TestCase):
    """
    Class to autoimaitc test Job class
    """
    def setUp(self):
        self.DURATION_TEST = 5
        self.START_TIME_TEST = "String"
        self.HARDWARE_TEST = "platform"
        self.BIN_FILE_TEST = "/tmp/falseBinary"
        self.TARGETS_TEST = "all"

        self.d = {"duration": self.DURATION_TEST,
                  "start_time": self.START_TIME_TEST,
                  "binaries": {
                      "hardware": self.HARDWARE_TEST,
                      "bin_file": self.BIN_FILE_TEST,
                      "targets": self.TARGETS_TEST
                  }
                  }

        self.falseBinary = Path(self.BIN_FILE_TEST)
        with self.falseBinary.open("w+") as fh:
            fh.write("Hello world!!")

        self._fileToCancel = []

    def tearDown(self):
        self.falseBinary.unlink()
        for f in self._fileToCancel:
            f.unlink()

    def _falseFile(self, filepath):
        with filepath.open('w+') as fh:
            fh.write(f"Hello world from {filepath}")
        self._fileToCancel.append(Path(filepath))

    def test_create_from_dict(self):
        j = Job(self.d)

        self.assertEqual(j.duration, self.DURATION_TEST)
        self.assertEqual(j.start_time, self.START_TIME_TEST)
        self.assertIsInstance(j.binaries, list)
        self.assertEqual(len(j.binaries), 1)
        self.assertIsInstance(j.binaries[0], Job.Binary)
        self.assertEqual(j.binaries[0].hardware, self.HARDWARE_TEST)
        self.assertEqual(j.binaries[0].bin_file, self.BIN_FILE_TEST)
        self.assertEqual(j.binaries[0].targets, self.TARGETS_TEST)

    def _test_validate_json(self, job, expected_error, expected_warnings):

        errors, warnings = job.validate()

        self.assertEqual(errors, expected_error)
        self.assertEqual(warnings, expected_warnings)

        # Binares field is handled in a list, returned string should see it
        if not isinstance(self.d['binaries'], list):
            self.d['binaries'] = [self.d['binaries'], ]

        # Targets in binaries object are always list,
        for b in self.d['binaries']:
            if not isinstance(b['targets'], list):
                b['targets'] = [b['targets'], ]

        self.assertLessEqual(job.toJSON(), json.dumps(self.d))

    def test_json(self):
        j = Job(self.d)

        # Binares field is handled in a list, returned string should see it
        self.d['binaries'] = [self.d['binaries'], ]

        self.assertLessEqual(j.toJSON(0), json.dumps(self.d))

    def test_not_existing_bin_file(self):

        NOT_EXISTING_PATH = "/not/existing/path"
        self.d['binaries']['bin_file'] = NOT_EXISTING_PATH

        j = Job(self.d)

        self._test_validate_json(j, [f"File {NOT_EXISTING_PATH} not found"],
                                 [])

    def test_start_time_values(self):
        # start_time can be a datetime value
        self.d['start_time'] = datetime.now()
        j = Job(self.d)

        self.d['start_time'] = self.d['start_time'].strftime("%Y-%m-%d %H:%M")

        self._test_validate_json(j, [], [])

    def test_legacy_ts_init(self):
        self.d['ts_init'] = self.d.pop('start_time')

        with self.assertWarnsRegex(FutureWarning, r"Deprecated use of .*"):
            j = Job(self.d)
            errors, warnings = j.validate()

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

        # Binares field is handled in a list, returned string should see it
        self.d['binaries'] = [self.d['binaries'], ]

        # Targets in binaries object are always list,
        for b in self.d['binaries']:
            if not isinstance(b['targets'], list):
                b['targets'] = [b['targets'], ]

        # TODO improve assert
        self.assertNotEqual(j.toJSON(), json.dumps(self.d))

    def test_legacy_image(self):
        self.d['image'] = self.d.pop('binaries')

        with self.assertWarnsRegex(FutureWarning, r"Deprecated use of .*"):
            j = Job(self.d)
            errors, warnings = j.validate()

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

        # Binares field is handled in a list, returned string should see it
        self.d['image'] = [self.d['image'], ]

        # Targets in image object are always list,
        for b in self.d['image']:
            if not isinstance(b['targets'], list):
                b['targets'] = [b['targets'], ]

        # TODO improve assert
        self.assertNotEqual(j.toJSON(), json.dumps(self.d))

    def test_legacy_python_script(self):
        falsePythonScript = '/tmp/pFile.py'
        self._falseFile(Path(falsePythonScript))

        self.d['python_script'] = falsePythonScript

        with self.assertWarnsRegex(FutureWarning, r"Deprecated use of .*"):
            j = Job(self.d)
            errors, warnings = j.validate()

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_orchestrator(self):
        orchestratorFile = '/tmp/orchestrator.py'
        self._falseFile(Path(orchestratorFile))

        self.d['orchestrator'] = {'file': orchestratorFile}

        j = Job(self.d)

        self._test_validate_json(j, [], [])

# IoTTestbed class-------------------------------------------------------------


class IoTTestbed:
    """
    Class that represent the testbed Frontend, it handles connection and
    operation
    """

    def __init__(self, token,  server, cacert=None, **kargs):

        if (not server.startswith("http://") and
            not server.startswith("https://")):
            raise ValueError("Server should start with http:// or https://")

        self.server = server
        self.token = token
        self.cacert = cacert if cacert is not None else True

        self._session = None

    def is_open(self):
        return self._session is not None

    def open(self):
        if self.is_open():
            # TODO: change error type
            raise Exception("Already connected")

        self._session = Session()
        self._session.headers = {
            'Authorization': f'Token {self.token}',
            'Accept': 'application/json'
        }
        if self.server.startswith("https"):
            if isinstance(self.cacert, Path):
                self._session.verify = str(self.cacert)
            else:
                self._session.verify = self.cacert

    def __enter__(self):
        if not self.is_open():
            self.open()
        return self

    def close(self):
        self._session.close()
        self._session = None

    def __exit__(self, type, value, traceback):
        self.close()

    def calendar(self, begin=datetime.now(),
                 end=datetime.now() + timedelta(days=7)):
        if not self.is_open():
            # TODO chaneg Excpetion type
            raise Exception("Not open")

        response = self._session.get(
            urljoin(self.server, '/calendar_data'),
            params={'begin__gte': begin, 'begin__lt': end}
        )

        if not response.ok:
            raise Exception(
                f"Something goes wrong: {response.status_code} "
                f"{response.json()}"
            )

        return response.json()

    def schedule(self, job):
        if not isinstance(job, Job):
            raise ValueError("job must be a Job object")

        errors, warnings = job.validate()

        if len(errors) > 0:
            raise Exception(f"Not valid job: errors {errors}")

        if not self.is_open():
            # TODO chaneg Exception type
            raise Exception("Not open")

        files = []
        for f in job.attachedFiles():
            print(f"attached: {f}")
            files.append(
                ('files', (f.name, f.open('rb'),
                           'application/octet-stream'))
            )

        r = self._session.post(urljoin(self.server, '/job/'),
                               data={'config': job.toJSON()},
                               files=files)

        for label, f in files:
            f[1].close()

        return r.status_code == 200, r.json()

    def cancel(self, jobId):
        if not isinstance(jobId, int):
            raise ValueError("jobId must be an int")

        if not self.is_open():
            # TODO change Exception type
            raise Exception("Not open")

        r = self._session.delete(
            urljoin(self.server, f"/job/{jobId}"))

        return r.ok, r.json()

    def _get_filename(self, tmp):
        if not tmp:
            return None

        fname = re.findall(r'filename="(.+)"', tmp)
        if len(fname) == 0:
            return None
        return fname[0]

    def download(self, jobId, destDir, delete=True, unzip=False):
        if not isinstance(jobId, int):
            raise TypeError("testId should be an integer")

        if isinstance(destDir, str):
            destDir = Path(destDir)
        elif not isinstance(destDir, Path):
            raise TypeError("destDir should be a str or Path object")

        if not destDir.is_dir():
            raise FileNotFoundError(
                f"{destDir} doesn't exist or is not a directory")

        if not self.is_open():
            raise Exception("Not open")

        r = self._session.get(
            urljoin(self.server, f"/completed/{jobId}"))

        if not r.ok:
            raise Exception("Error downloading completed job file: "
                            f"{r.status_code} {r.content}")

        filename = self._get_filename(r.headers.get('content-disposition'))
        if not filename:
            filename = f"job_{jobId}.tar.gz"

        cLen = r.headers.get('content-length')
        if cLen is None:
            raise Exception('Received data does not contain expected content-length')

        cLen = int(cLen)

        if cLen != len(r.content):
            raise Exception("Received data is of different length than expected "
                            f"(received: {len(r.content)} expected: {cLen})")

        fpath = destDir.joinpath(filename)
        count = fpath.write_bytes(r.content)

        if delete:
            r = self._session.delete(
                urljoin(self.server, f"/completed/{jobId}"))
            if not r.ok:
                raise Exception("Error deleteing downloaded file on server "
                                f"{r.status_code} {r.content}")

        if unzip:
            try:
                shutil.unpack_archive(fpath, destDir, format='gztar')
            except Exception as e:
                print(
                    "Warning: failed to extract the downloaded log "
                    f"{filename} to {destDir}"
                )
                print(e)

        return str(fpath), count

    def list_completed(self):
        if not self.is_open():
            raise Exception("Not open")

        r = self._session.get(urljoin(self.server, "/completed/"))

        if not r.ok:
            raise Exception(
                f"Error retrivieng completed list: {r.status_code} {r.content}"
            )

        return r.json()

    def add_reservation(self, island, begin, end):
        if not isinstance(begin, datetime):
            raise ValueError("Begin argument is not valid")

        if not isinstance(end, (datetime, timedelta)):
            raise ValueError("End argument is not valid")

        if isinstance(end, timedelta):
            end = begin + end

        if not self.is_open():
            raise Exception("Not open")

        r = self._session.post(
            urljoin(self.server, "/reservations/"),
            data={'island': island, 'begin': begin, 'end': end}
        )

        if not r.ok:
            raise ApiError(
                contextMessage="Error adding reservation",
                code=r.status_code,
                message=r.json()
            )

        return r.json()

    def mod_reservation(self, reserv_id, island, begin, end):
        if not isinstance(reserv_id, int):
            raise ValueError("reserv_id should be an integer id")

        if begin is not None and not isinstance(begin, (datetime, timedelta)):
            raise ValueError("Begin argument is not valid")

        if end is not None and not isinstance(end, (datetime, timedelta)):
            raise ValueError("End argument is not valid")

        if not self.is_open():
            raise Exception("Not open")

        old_response = self._session.get(
            urljoin(self.server,
                    f"/reservations/{reserv_id}"
                    )
        )

        if not old_response.ok:
            raise ApiError(
                contextMessage="Error retrieving reservation to modify it",
                code=old_response.status_code,
                message=old_response.json()
                )

        print(f"{old_response.json()}")
        old = old_response.json()

        new = {'island': island if island is not None else old['island']}

        if begin is None:
            new['begin'] = old['begin']
        elif isinstance(begin, timedelta):
            new['begin'] = datetime.fromisoformat(old['begin']) + begin
        else:
            new['begin'] = begin

        if end is None:
            new['end'] = old['end']
        elif isinstance(end, timedelta):
            new['end'] = datetime.fromisoformat(old['end']) + end
        else:
            new['end'] = end

        r = self._session.put(
            urljoin(self.server, f"/reservations/{reserv_id}"),
            data=new
        )

        if not r.ok:
            raise ApiError(
                contextMessage=f"Error modifying the reservation {reserv_id}",
                code=r.status_code,
                message=r.json()
                )

        return r.json()

    def del_reservartion(self, reserv_id):
        if not isinstance(reserv_id, int):
            raise ValueError("reserv_id should be an integer id")

        if not self.is_open():
            raise Exception("Not open")

        r = self._session.delete(
            urljoin(self.server, f"/reservations/{reserv_id}")
        )

        if not r.ok:
            raise ApiError(
                contextMessage="Error deleting reservation",
                code=r.status_code,
                message=r.json()
            )

        return r.json()


def validate(jobFile):
    print(f"File: {jobFile}")

    if not isinstance(jobFile, (str, Path)):
        raise TypeError("jobFile should be a str or Path object")

    if isinstance(jobFile, str):
        jobFile = Path(jobFile)

    with jobFile.open('r') as fh:
        job = Job(json.load(fh), jobFile.parent)

    errs, warns = job.validate()

    if len(errs) > 0:
        print("Errors:")
        for e in errs:
            print(f"\t{e}")

    if len(warns) > 0:
        print("Warnings:")
        for w in warns:
            print(f"\t{w}")

    if len(warns) == 0 and len(errs) == 0:
        print("Job file valid")
    elif len(errs) != 0:
        print("Job file has error, rejected")
    elif len(warns) != 0:
        print("Job file has warings, accepted")

    return job, errs, warns


def parse_datetime_argument(arg):
    if not isinstance(arg, str):
        raise ValueError("arg should be a string")

    if arg == 'now':
        return datetime.now() + timedelta(seconds=20)

    m = re.match(r" ?(?P<number>[+-]\d+)(?P<unit>[mh])", arg)
    if m:
        if m.group('unit') == 'm':
            return timedelta(minutes=int(m.group('number')))
        elif m.group('unit') == 'h':
            return timedelta(hours=int(m.group('number')))

    try:
        return datetime.fromisoformat(arg)
    except Exception as err:
        raise ValueError(f"Can't parse reservation argument '{arg}'") from err


class TestParseReservationArgument(unittest.TestCase):
    def test_int(self):
        with self.assertRaisesRegex(ValueError, "arg should be a string"):
            parse_datetime_argument(1)

    def test_str(self):
        for value in ['+string', 'ciao']:
            with self.assertRaises(ValueError):
                parse_datetime_argument(value)

    def test_now(self):
        r = parse_datetime_argument('now')

        self.assertTrue(isinstance(r, datetime))
        self.assertLess(datetime.now() - r, timedelta(seconds=1))

    def test_relative(self):
        values = [
            ('-1h', timedelta(hours=-1)),
            (' -1h', timedelta(hours=-1)),
            ('-1m', timedelta(minutes=-1)),
            (' -1m', timedelta(minutes=-1)),
            ('+1m', timedelta(minutes=1)),
            ('+1h', timedelta(hours=1)),
        ]
        for s, v in values:
            with self.subTest(s=s):
                r = parse_datetime_argument(s)

                self.assertTrue(isinstance(r, timedelta))
                self.assertEqual(r, v)

    def test_absoulte(self):
        values = ['2021-02-07 12:00', '2021-02-07T12:00']
        ref = datetime(year=2021, month=2, day=7, hour=12, minute=0)
        for v in values:
            with self.subTest(v=v):
                r = parse_datetime_argument(v)

                self.assertTrue(isinstance(r, datetime))
                self.assertEqual(r, ref)


if __name__ == '__main__':

    success, config = Config.load( { }, Path.cwd(), Path.home(), default_config )

    import argparse
    cli_parser = argparse.ArgumentParser(
        description="Client script for testbed Iot"
    )
    subparser = cli_parser.add_subparsers(dest='command')

    cli_parser.add_argument('--server', help="URL of testbed server")
    cli_parser.add_argument('--token', '-t', help="Token used to authenticate")
    cli_parser.add_argument('--cacert',
                            help="indicate which CA certificate use to verify "
                            "server certificate"
                            )

    cli_parser.add_argument('--hook', dest='hook',
                            type=str,
                            nargs='+',
                            default=[],
                            help="call the main function of this file at the end of the schedule operation")

    cli_parser.add_argument('--clear-hooks', dest='hook',
                            action='append_const',
                            const=None,
                            help="call the main function of this file at the end of the schedule operation")

    def parse_pairs(a: str):
        if '=' not in a:
            raise ValueError("Pairs of hook-data should be field=value")

        return a.split('=', maxsplit=1)

    cli_parser.add_argument('--hook-data', 
                            dest='hook_data',
                            type=parse_pairs,
                            nargs='*',
                            default=[],
                            help="data passed to all the hooks")

    # Save config subcommand --------------------------------------------------
    saveConfig = subparser.add_parser(
        'saveConfig',
        help="save configuration as default for user"
    )

    # Show config subcommand --------------------------------------------------
    showConfig = subparser.add_parser(
        'showConfig',
        help="show the active configuration"
    )

    # Calendar subcommand --------------------------------------------------
    calendarParser = subparser.add_parser(
        'calendar',
        help='interrogate testbed to know when the testebd will used, by '
        'default show next 7 days'
    )
    calendarParser.add_argument(
        '--begin', '-b', nargs='?',
        default=datetime.combine(date.today(), time()) - timedelta(hours=12),
        help="Request all the reservation that began after the specified "
        "moment. Valid values are: a date in ISO format YYYY-mm-dd, "
        "a quoted date and time in ISO format 'YYY-mm-dd HH:MM' or an "
        "empty value for whenever. By default, when this option is "
        "not used, is yesterday at noon."
    )
    calendarParser.add_argument(
        '--end', '-e', nargs='?',
        default=datetime.combine(date.today(), time()) + timedelta(days=7),
        help="Request all the reservation that ends before the specified "
        "moment. Valid values are: a date in ISO format YYYY-mm-dd, a "
        "quoted date and time in ISO format 'YYY-mm-dd HH:MM' or an empty "
        "value for whenever. By default, when this option is not used, is "
        "in 7 days at midnight."
    )

    calendarParser.add_argument(
        'island',
        choices=ISLAND_LIST,
        nargs='?',
        default=config.calendar_island, 
        help="Specify to show a single or all islands"
    )

    # Validate subcommand -----------------------------------------------------
    validateParser = subparser.add_parser(
        'validate',
        help='validate a jobFile only on client side, server will do a '
        'better and deeper check before accept the job'
    )
    validateParser.add_argument('jobFile',
                                help='the json file that decribe the job',
                                nargs='+')

    # Scheduler subcommand ----------------------------------------------------
    scheduleParser = subparser.add_parser(
        'schedule',
        help='perform a validation as validate command and if '
        'it is passed the test is submitted to Testbed'
    )
    startGroup = scheduleParser.add_mutually_exclusive_group()
    startGroup.add_argument('--asap', action="store_true",
                            help="overwrite start_time fields with 'asap'")
    startGroup.add_argument(
        '--asap-after', metavar="AFTER",
        help="overwrite start_time fields with 'asap AFTER', AFTER should be "
        "in form (quoted) 'YYYY-mm-dd HH:MM' or in form (quoted or not) HH:MM"
    )
    startGroup.add_argument('--now', action='store_true',
                            help="overwrite start_time fields with 'now'")

    scheduleParser.add_argument('--duration', type=int,
                                default=None,
                                help="overwrite the duration of the job (in seconds)")

    scheduleParser.add_argument('jobFile',
                                help='the json file that decribe the job',
                                nargs='+')

    # Cancel subcommand --------------------------------------------------
    cancelParser = subparser.add_parser('cancel')
    cancelParser.add_argument('jobId', help="ids of jobs to cancel",
                              type=int, nargs='+')

    # Download subcommand --------------------------------------------------
    downloadParser = subparser.add_parser('download',
                                          help='downalod completed')
    downloadParser.add_argument('jobId', help="ids of jobs to download",
                                type=int, nargs='*')
    downloadParser.add_argument('--unzip', '-u',
                                help="unzip after downloading",
                                action='store_true')
    downloadParser.add_argument(
        '--dest-dir',
        help="specify in which directory download files",
        default=None
    )
    downloadParser.add_argument(
        '--no-delete', action='store_false',
        help="avoid delete file on server after it is downloaded"
    )

    # list_completed subcommand -----------------------------------------------
    list_completedParser = subparser.add_parser(
        'completed',
        help="retrieve list of user's completed job"
    )

    # reservation subcommand --------------------------------------------------
    reservationParser = subparser.add_parser(
        "reservation",
        help="add, change or delete a reservation"
    )
    reservationCommand = reservationParser.add_subparsers(dest='reserv_cmd')

    reserv_add_parser = reservationCommand.add_parser('add',
                                                      help='Add a reservation')

    reserv_add_parser.add_argument(
        'island',
        help='name of island to reserve'
    )

    reserv_add_parser.add_argument(
        'begin',
        help='indicate when the reservation begins, this should be a '
        'datetime in ISO format YYYY-mm-ddTHH:MM, or a quote string '
        '"YYYY-mm-dd HH:MM" or special value "now" (this means in 20s '
        'to have time to prepare and submit the request)'
    )
    reserv_add_parser.add_argument(
        'end',
        help='indicate when the reservation ends. This accept same '
        'format of begin plus a relative form: +Nm/+Nh where N is '
        'a number and m means minutes while h means hour'
    )

    reserv_mod_parser = reservationCommand.add_parser(
        'mod', help='Modify a reservation'
    )
    reserv_mod_parser.add_argument('id', type=int,
                                   help='ID of reservation you want modify')

    reserv_mod_parser.add_argument(
        "--island", '-i',
        help="new island for reservation"
    )

    reserv_mod_parser.add_argument(
        '--begin', '-b',
        help="indicate the begin time changes, it can be an absolute "
        "value (expressed as ISO format or quotes string) or special value "
        "now or a relative value in one of forms: ' -Nh', ' -Nm', +Nm, +Nh "
        "where N is a number and m means minutes and h means hours.\n"
        "NOTE negative value should be in quoted string that start with a "
        "space"
    )

    reserv_mod_parser.add_argument(
        '--end', '-e',
        help="indicate the end time changes, it can be an absolute "
        "value (expressed as ISO format or quotes string) or special "
        "value now or a relative value in one of forms: ' -Nh', ' -Nm', "
        "+Nm, +Nh where N is a number and m means minutes and h means "
        "hours.\nNOTE negative value should be a quoted string that start "
        "with a space"
    )

    reserv_del_parser = reservationCommand.add_parser(
        'del', help='Delete a reservation'
    )
    reserv_del_parser.add_argument('id', type=int,
                                   help='ID of reservation you want modify')

    args = cli_parser.parse_args()

    success, config = Config.load(
        {
            k: getattr(args, k) 
            for k in ['server', 'cacert', 'token'] 
            if hasattr(args, k) and getattr(args, k) is not None
        },
        Path.cwd(),
        Path.home(),
        default_config
    )

    if not success:
        raise ValueError("Error in the configuration syntax or values")

    if args.command == 'calendar':
        if args.begin is not None and isinstance(args.begin, str):
            try:
                args.begin = datetime.fromisoformat(args.begin)
            except Exception:
                print("Can't parse begin argument: wrong format")
                raise

        if args.end is not None and isinstance(args.end, str):
            try:
                args.end = datetime.fromisoformat(args.end)
            except Exception:
                print("Can't parse end argument: wrong format")
                raise

    if args.command == 'reservation':
        if args.reserv_cmd == 'mod' and args.begin is None and \
           args.end is None and args.island is None:
            print("To modify a reservation at least one of island"
                  ", begin or end should be changed ")
            exit(1)

        if args.reserv_cmd in ['add', 'mod']:
            if args.begin is not None:
                args.begin = parse_datetime_argument(args.begin)

            if args.end is not None:
                args.end = parse_datetime_argument(args.end)

    if args.command == 'saveConfig':
        with configFile.open('w') as fh:
            json.dump(dict(config), fh, default=str, indent=1)
        configFile.chmod(0o700)
        exit(0)

    if args.command == 'showConfig':
        print('The current active config is:')
        pprint.pprint(dict(config))
        exit(0)

    if args.command == 'validate':
        for tfile in args.jobFile:
            validate(tfile)
            print('---------------')

        exit(0)

    if args.command == 'download':
        if (args.dest_dir is None):
            args.dest_dir = config.dest_dir

        if not Path(args.dest_dir).is_dir():
            print(f"Destination directory '{args.dest_dir}' not found")
            exit(1)

    # We use None as a signal to find clear the list of hooks, so we want to
    # find the last None and delete every hook that comes before it

    # First get the complete list of hooks
    args.hook = [*config.hooks.get(args.command,[]), *args.hook]
    # Then if there is a None in the list we reverse it to find the last one, 
    # we get the index in the original list (len - idx of reverse list), and
    # then we use that to filter every hook that comes before it 
    # (included the None)
    if None in args.hook:
        args.hook = args.hook[len(args.hook)-args.hook[::-1].index(None):]

    args.hook_data = dict(args.hook_data)

    if args.command == 'schedule' and args.asap_after:
        m = re.match(r'(\d{4}-\d{2}-\d{2}\s+)?\d{2}:\d{2}', args.asap_after)
        if m is None:
            print(f"--asap-after argument '{args.asap_after}' is not in "
                  "form [YYYY-mm-dd] HH:MM")
            exit(1)

    if config.token is None:
        print("Token is required to authenticate on testbed")
        exit(1)

    with IoTTestbed(token=config.token, server=config.server, cacert=config.cacert) as tiot:
        tiot.schedule = add_hooks(tiot.schedule, args.hook, args.hook_data)
        tiot.download = add_hooks(tiot.download, args.hook, args.hook_data)

        if args.command == 'calendar':
            try:
                busyList = tiot.calendar(args.begin, args.end)
            except Exception:
                raise
            else:
                if args.island != 'all':
                    busyList = [busy for busy in busyList if busy['island']['name'] == args.island]

                Formatter.format_calendar(busyList, show_island=(args.island == 'all'))

            exit(0)

        elif args.command == 'schedule':
            for tfile in args.jobFile:
                job, errs, warns = validate(tfile)

                if args.asap:
                    job.start_time = 'asap'

                if args.asap_after:
                    job.start_time = f'asap {args.asap_after}'

                if args.now:
                    job.start_time = 'now'

                if args.duration is not None:
                    job.duration = args.duration

                if len(errs) == 0:
                    print("Submit job to testbed")
                    result = tiot.schedule(job)
                    print(f"result: {result}")

        elif args.command == 'cancel':
            for jid in args.jobId:
                b, result = tiot.cancel(jid)
                msg = f"Job {result['job_id']} {result['result']}"
                if result['scheduled']:
                    msg += f" at {result['scheduled'][1]}"
                print(msg)

        elif args.command == 'completed':
            print("Completed job:")
            for j in tiot.list_completed():
                print(f"\tId: {j['id']} completed on {j['end']}")

        elif args.command == 'download':
            if not args.jobId:
                args.jobId = [x['id'] for x in tiot.list_completed()]

            for id in args.jobId:
                try:
                    print(f"Download test {id} to {args.dest_dir}")
                    fname, nBytes = tiot.download(id, args.dest_dir,
                                                  args.no_delete, args.unzip)
                    formatted_bytes = Formatter.format_size(nBytes)
                    print(f"Saved {formatted_bytes} in {fname}")
                except Exception as ex:
                    print(f'Could not download the {id} job')
                    print(ex)

        elif args.command == 'reservation':
            try:
                if args.reserv_cmd == 'add':
                    r = tiot.add_reservation(args.island, args.begin, args.end)
                    print(f"added reservation with id {r['id']} since "
                          f"{r['begin']} to {r['end']}")

                elif args.reserv_cmd == 'mod':
                    r = tiot.mod_reservation(args.id, args.island, args.begin,
                                             args.end)
                    print(f"Resevation {r['id']} has been modified")

                elif args.reserv_cmd == 'del':
                    r = tiot.del_reservartion(args.id)
                    print(f"Resevation {r['id']} has been cancelled")

            except ApiError as ex:
                print(ex.contextMessage)
                for err in ex.message.values():
                    if isinstance(err, list):
                        print("\t".join(err))
                    else:
                        print(f"\t{err}")
