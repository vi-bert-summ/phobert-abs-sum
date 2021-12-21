import glob
import datasets
import pandas as pd
import concurrent.futures


def listPaths(path):
    """
    Input: Path of folder file 
    Output: list pathfile 
    """
    pathfiles = list()
    for pathfile in glob.glob(path):
        pathfiles.append(pathfile)
    return pathfiles


def read_content(pathfile):
    """
    Input: Path of txt file
    Output: A dictionary has keys 'original' and 'summary'
    """
    with open(pathfile) as f:
        rows  = f.readlines()
        original = ' '.join(''.join(rows[4:]).split('\n'))
        summary = ' '.join(rows[2].split('\n'))
            
    return {'file' : pathfile,
            'original': original, 
            'summary': summary}


def get_dataframe(pathfiles):
    """
    Input: Path of txt file
    Output: DataFrame path, title, summary
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = executor.map(read_content, pathfiles)

    # Make blank dataframe
    data_df = list()
    for d in data:
        data_df.append(d)
    data_df = pd.DataFrame(data_df)
    data_df.dropna(inplace=True)
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    return data_df


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text