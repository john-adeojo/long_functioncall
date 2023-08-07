class SingletonToken:
    """
    Implements a singleton token.

    This class is used to implement a singleton token which can be set and retrieved using the 
    set_token() and get_token() class methods respectively. The token is stored in a private class 
    variable and can be accessed across different instances of the class.
    """
    __token = None

    @classmethod
    def set_token(cls, token):
        cls.__token = token

    @classmethod
    def get_token(cls):
        return cls.__token