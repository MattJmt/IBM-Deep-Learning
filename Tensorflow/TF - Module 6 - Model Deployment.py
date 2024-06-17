#Downloading Flask App
#Once you trained and saved your model, you want to deploy it on the web and provide a graphic user interface for people to interact with it.
# Flask is a lightweight web framework that allows us to host and deploy our machine learning model.
#The code below downloads and unzip the Flask app.
!pip install wget
import wget, zipfile

#source_zipfile='Python-Flask-MNIST-sample-app.zip'
filename='Python-Flask-MNIST-sample-app'

if not os.path.isfile(filename):
    filename = wget.download('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Flask-App/Python-Flask-MNIST-sample-app.zip')
    with zipfile.ZipFile("Python-Flask-MNIST-sample-app.zip","r") as zip_ref:
        zip_ref.extractall()
# In order for your app to access your machine learning model we need the model_deployment_endpoint_url of your model.
# First we install Watson Machine Learning package by running the code below.
!bx plugin install machine-learning

# Then we need to list out which instance of our saved machine learning model we want to deploy.
#
# To do that we need to set the environment variables for Watson Machine Learning
#
# To access Watson Machine Learning
#
# Log into IBM Cloud ('https://cloud.ibm.com/login').(Takes you to your IBM Cloud dashboard.
# In your IBM Cloud dashboard, click the Watson Machine Learning service instance for which you want to retrieve credentials. (This opens the service details page for the Watson Machine Learning service instance.) Click Service credentials.
# If there are no service credentials yet, click the New credential button.
# Under the ACTION menu, click "View credentials".
# Then Set
#
# ML_ENV=value of url
# ML_USERNAME=value of username
# ML_PASSWORD=value of password
# ML_INSTANCE=value of instance_id
# Run the code below to list your saved machine learning model instances.
!bx plugin show machine-learning
%%bash
export ML_ENV=
export ML_USERNAME=
export ML_PASSWORD=
export ML_INSTANCE=
bx ml list instances
# In the last line in the code cell below,
# bx ml show deployments model_id deployment_id
# replace model_id with "Model Id", and replace deployment_id with "Deployment Id" from the model in the output above.
# To get the model_deployment_endpoint_url of your model run the code below with the same environment variables as the previous code cell.
# Get your GUID and replace GUID below.
%%bash
bx ml set instance GUID
bx ml list deployments
%%bash
bx ml show deployments model_id deployment_id
# Scoring endpoint will be your model_deployment_endpoint_url

#In order to deploy your access your saved model, we need to give your app the url of your saved model.
#In the folder of your Flask app there is a file called server.py. Fill in model_deployment_endpoint_url with the "Scroing endpoint" of your model.
#And also fill in your wml_credentials

# Create a Python Cloud Foundry app here ('https://cloud.ibm.com/catalog#services'): Create a Python Cloud Foundry app
# Give the app a unique name (for the rest of this example, the sample name will be: "your-name-machine-learning-app")
# Accept the defaults in the other fields of the form
# Choose the 128 MB plan
# Click Create
# The Visit App URL contains the link to the app
# In the folder of your Flask app there is a file called manifest.yml. Replace app-name with the name of your app. And make sure the memory is set to "128mb".
# In another file named setup.py replace app-name with the name of your app.
wml_credentials = {
  "instance_id": "",
  "password": "",
  "url": "",
  "username": ""
}

# In the code cell below, replace email with you email that you use for your IBM Cloud Log in, and password with your IBM Cloud Log in password.
# Replace region with a number from 1 to 6.
# au-syd
# jp-tok
# 3. eu-de
# eu-gb
# us-south
# us-east
# Then run the code below to push your app onto IBM cloud.
%%bash
cd Python-Flask-MNIST-sample-app/app
ibmcloud login
email
password
'region'
region
N
ibmcloud target --cf
ibmcloud app push
# The Visit App URL link in cloud foundry contains the url to your Machine Learning App
# If you forget the url takes the form of "https://[app-name].mybluemix.net".