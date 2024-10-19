import pandas as pd
import numpy as np

class Ragify:
    def __init__(self,
                 llm_name: str,
                 ):
        self.llm_name = llm_name

    def generate_response(self,
                          question: str
                          ):
        return question