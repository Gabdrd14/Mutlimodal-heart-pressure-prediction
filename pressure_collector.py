import wfdb
import matplotlib.pyplot as plt
import pywt
from preprocessing import DataLoaderPreprocessFile


class CleanPreprocessingPipeline:
    def __init__(self, record_path, method="raw"):
        self.loader = DataLoaderPreprocessFile(record_path)
        self.data = None   

    def run(self):
        data = self.loader.load()

        RHC_pressure = data["RHC_pressure"]
        patch_Pre = data["patch_Pre"]

        # print(data.keys())


        self.data =  {

            
            "RHC_pressure": RHC_pressure,
            "patch_Pre": patch_Pre,
        }

        return self.data
    
    def getValue(self):

        return self.data


INPUT = 'dat_signals/TRM107-RHC1'


pipeline = CleanPreprocessingPipeline(INPUT)


a = pipeline.run()



value  = a["RHC_pressure"]


plt.plot(value)

plt.show()