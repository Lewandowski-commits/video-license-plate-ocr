def get_database():
    from os import environ
    from pymongo import MongoClient
    USER = environ['MONGODB_USER']
    PASSWORD = environ['MONGODB_PASSWORD']

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = f'mongodb+srv://{USER}:{PASSWORD}@cluster0.i1i94.mongodb.net/myFirstDatabase?retryWrites=true&w=majority'

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client['license_plates']


def verify_dtypes(*items):
    '''
    Takes in a tuple of arguments and checks if the elements are dicts. If not, returns an exception, otherwise returns the same list.
    '''
    # raise an exception if the passed in arguments are not dicts
    dtypes = [False if not type(item) == dict else True for item in items]

    if not all(dtypes):
        raise TypeError('Passed in arguments must be dicts.')

    return items


# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":
    # Get the database
    dbname = get_database()
    collectionname = dbname['filename']

