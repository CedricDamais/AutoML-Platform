class InternalServerError(Exception):
    def __init__(self, message="Internal Server Error", code=500):
        self.message = message
        self.code = code
        super().__init__(self.message)