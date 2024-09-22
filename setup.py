from setuptools import find_packages,setup
HYPHEN_E_DOT='-e.'

def get_requirements(path):
    l=[]
    with open(path) as f:
        l=f.readlines()
        l=[x.replace('\n','') for x in l]   
        if HYPHEN_E_DOT in l:
            l.remove(HYPHEN_E_DOT)

    return l


setup(
    name='ML PROJECT WORKFLOW',
    version='0.0.1',
    author='Sameer',
    author_email='sameer.smiley123@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)