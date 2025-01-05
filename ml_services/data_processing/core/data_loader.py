from .data_class import CustomDataFrame

from typing import List, Dict, Tuple, Optional

class DataLoader():
    def __init__ (self):
        pass
    
    def load(self, file: Optional[str], file_type: Optional[str], connection_string: Optional[str], connection_type: Optional[str]):
        pass
    
    
class WebLoader(DataLoader):
    
    def load(self, file: Optional[str], file_type: Optional[str], connection_string: Optional[str], connection_type: Optional[str]):
        pass
    
class FileLoader(DataLoader):
    
    def load(self, file: Optional[str], file_type: Optional[str], connection_string: Optional[str], connection_type: Optional[str]):
        pass
    
class DBLoader(DataLoader):
    
    def load(self, file: Optional[str], file_type: Optional[str], connection_string: Optional[str], connection_type: Optional[str]):
        pass
    
class Loader():
    
    def __init__ (self, file: Optional[str], file_type: Optional[str], connection_string: Optional[str], connection_type: Optional[str]):
        self.file = file
        self.f_type = file_type
        self.uri = connection_string
        self.u_type = connection_type
        self.loader = None
        
    def load_data(self) -> CustomDataFrame:
        
        if self.file:
            self.loader = FileLoader()
        
        elif self.uri:
            if self.u_type in ['api', 'web']:
                self.loader = WebLoader()
                
            elif self.utype == 'db':
                self.loader = DBLoader()
                
            else:
                raise ValueError("Connection type not specfied, please provide type of connection for data retrieval")
        
        
        
        try:
            data:CustomDataFrame = loader.load(self.file, self.f_type, self.uri, self.u_type)
            return data
            
        except:
            print('Unable to load data')
            