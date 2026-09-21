import os
import pickle
import data
import shutil


class DataCacher:

    def __init__(self):
        self.data = data.Data()
        self._cache_dir = "_cache"
        os.makedirs(self._cache_dir, exist_ok=True)
        self._create_cache()

    def _create_cache(self):
        files = os.listdir(self._cache_dir)
        for function_name in dir(self.data):
            if function_name.startswith("get_"):
                if "%s.bin" % function_name not in files:
                    pickle.dump(getattr(self.data, function_name)(),
                                open("./%s/%s.bin" % (self._cache_dir, function_name), "wb"))
                # replace function in data class with cached version, use double-lambda to create copy of input var:
                setattr(self.data, function_name,
                        (lambda x: lambda: pickle.load(open("./%s/%s.bin" % (self._cache_dir, x), "rb")))(function_name))

    def rebuild_cache(self):
        shutil.rmtree(self._cache_dir)
        self.__init__()
