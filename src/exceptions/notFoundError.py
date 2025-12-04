class DataSetNotFoundError(Exception):
    def __init__(self, message="Resource Not Found", code=404):
        self.message = message
        self.code = code
        super().__init__(self.message)