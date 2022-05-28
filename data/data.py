import pandas as pd

class TextToCsv:
    def __init__(self, en_file, np_file, target_file) -> None:
        self._en_file = en_file
        self._np_file = np_file
        self._target_file = target_file
        self._write_to_csv_file()
    
    def _read_file(self, file):
        with open(file, 'r', encoding='utf-8') as file:
            data = file.read()
        return data.split('\n')
    
    def _combine_file(self):
        en_text = self._read_file(self._en_file)
        np_text = self._read_file(self._np_file)
        data = zip(np_text, en_text)
        df = pd.DataFrame(data, columns=['Nepali', 'English'])
        return df
    
    def _write_to_csv_file(self):
        df = self._combine_file()
        df[df.columns] = df.apply(lambda x: x.str.strip())
        df.to_csv(self._target_file, index=False)