## My Project <img align="right" width="120" height="120" src="/assets/IMG/workoutstats.jpg">
In this project, I applied machine learning techniques in order to determine how the variables linked with lifestyle and workout may be utilized to forecast the number of calories burned throughout the exercise periods. The report outlined below details my method of approach and findings.

***

## Introduction
Proper calculation of the calorie burned in the exercises would be vital in planning exercises, weight loss and individual health evaluation. Simple kinds of estimation like formulas or algorithms in a fitness tracker do not turn out to be very exact when it comes to human physiology. The amount of calories burned will add to several variables, such as the rate of the heartbeat, exercise length, body structure, fitness and kind of exercise taken. Considering the complexities, I sought to come up with a more accurate and data-oriented model that happens to be machine learning-based.

The data that I have worked with consists of approximately 20,000 workout sessions, each being represented in terms of biometric values, heart rate data, type of workout, fitness attributes and total caloric expenditure. The target variable, Calories_Burned is a numerical variable that was counted on each example; hence a supervised regression technique was suitable. I was aiming to assess the efficiency of ecstasis of burning in terms of predicting the number of calories required basing only on a lifestyle and no-fitness-related variables, and to compare the use of which machine learning model. My experiment with a variety of models after running them revealed that Linear Regression and Gradient Boosting gave the most accurate results.

***

## Data
The initial sample was composed of demographic information, nutritional facts, descriptions on exercise, body composition data, and other lifestyle factors. Given that the aim of this project was to predict how many calories are burned during working out and physiologic factors, any type of variables regarding diets (e.g. names of meals, their content of macronutrients, time needed to prepare it, and nutrient content) were not considered. This filtering meant that the analysis would focus on exercise-based predictors, such as heart rate indicators, the type of work-out, how long the workout would last, the percentage of body fat, and the level of fitness experience. The dataset itself was preprocessed, and after it, it consisted of 38 pertinent columns and approximately 20,000 entries.

I started by performing preprocessing tasks that included: 1) finding numeric features, 2) filling-in the gaps with the means of columns, and 3) standardising the data with the StandardScaler that implemented the job of normalising the input distributions. I then divided the data into sets of training and test at 80/20. In the context of my exploratory data analysis, I have created a correlation heatmap, in which it occurred that the strongest predictors of calories burned were maximum and average BPM, and session duration. On the other hand, there was weak or no correlation between demographics like age and height.

Figure Caption:
This heatmap indicates correlation of numeric features. There is a strong positive correlation between heart rate indicators and time spent 
working out as well as calories burned, whereas such static characteristics as age do not seem to have any significant impact.

***

## Modelling
As a method to model the calorie expenditure, I applied four supervised regression models, namely, Linear Regression, Random Forest, Gradient Boosting, and XGBoost. Linear Regression was taken as a baseline model to estimate the linear terms of features and calorie output. More elaborated, non-linear interactions were modeled using random forest. Gradient Boosting also enhanced predictions by refining on residual errors thus it is appropriate to data that is moderately non-linear. XGBoost implemented the optimal boosting structure to achieve the additional rapidity and precision.

All the models were trained on the standardized training data, and the test set was predicted. Below is an example of a code that is used to train the Gradient Boosting model.
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

```

***

## Results
I also checked the models on Matthean Squared Error (MSE), root mean squared error (RMSE), and the coefficient of determination (R2). Linear Regression achieved the best output of a R2 of 0.6666 and RMSE of 288.36. Close behind was Gradient Boosting with an R 2 of 0.6622. Conversely, the simpler Linear Regression and more focused Gradient Boosting algorithm did the simplest and more specific operation and thus were more productive in this task than the more complex ensemble procedures such as Random Forest (R2 of 0.6519) and XGBoost (R2 of 0.6469).

The scatter plots were made to compare the actual versus model prediction of calorie in the models. Such plots demonstrate the proximity of the approximations with the actual results.

The plots of feature importance were created, always showing the predictors of calorie burn as the maximum BPM, average BPM, and session duration. Such findings are consistent with general physiological laws, according to which the intensity and the duration of an exercise are the key determinants of energy utilization.

The gradient boosting model also sketched a learning curve, which was used to determine how the training performance improved with each iteration and also to ensure possible overfitting.

***

## Discussion
Linear Regression and Gradient Boosting have been calculated using models and model results indicate that 2/3 of the calorie burn variance was accounted. The good performance of Linear Regression models indicates that the predominant data trends were linear, whereas Gradient Boosting precision was an indicator of minute non-linear trends. There are however some limitations to this analysis. The model itself has been trained using the two 20,000 available workout sessions and this may not reflect the entire population groups and conditions of exercise. Notably important variables like the continuous changes in heart rate, the environment and omission of dietary factors, may constrain the capacity of the model to encompass all the effects on calories expenditure. Also, outcomes can be affected by the quality of the data and the accuracy of self-reported or variables measured by devices. Irrespective of these drawbacks, as experienced in the analysis, machine learning can be utilized to generate meaningful predictions using available fitness data in the field of exercise science. These results prompt additional research of finer physiological outcomes and higher volume and more varied datasets to increase accuracy and generalizability of a model.

Scatter plots of the actual and the predicted values of Linear Regression and Gradient Boosting revealed that actual calorie values were very close to those of the actual calorial values as observed by the clustering of the values around the diagonal. Random Forest and XGBoost exhibited more scatter and less alignment, which proves that their higher level of complexity did not provide more accurate results than the most successful ones. The importance of heart rates measurements and time spent in the session supported the central role of these variables and confirmed the previously known correlation between intensity, duration, and calories burned on a workout.

***

## Conclusion
Overall, the best predictors of calories burned based on workout and lifestyle data were Linear Regression and Gradient Boosting, which indicated the influence of the heart rate and duration of the workout. Measures of accuracy were ensured by excluding irrelevant variables.

Continuous heart-rate data, deep learning model, and hyperparameter tuning could be used in the future, or the dataset could be expanded to enhance the generalizability.
















**Hi class, welcome to the AOS C111/204 final project!** <img align="right" width="220" height="220" src="/assets/IMG/template_logo.png">

For this project, you will be applying your skills to train a machine learning model using real-world data, then publishing a report on your own website.

* To get data for your project, you could:
  * use **your own data** from a separate research activity
  * **scour the internet** to find something original, then preprocess it yourself - see the Module Overview on BruinLearn for some resources
  * browse an archive of data designed for machine learning problems, such as the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/datasets)

* Your report should be written using a scientific structure. [This template page](/project.md) gives an example structure that you could use, but feel free to make it your own. See Bruinlearn for some examples from previous students.
* To get high marks: **apply things that you learnt about in class**, and **explain your process in the report**, i.e. why you thought this would be interesting, why you decided to use a particular model, the challenges that you faced processing your data, etc.

Your website will be a great addition to your CV, and a place to host future projects too since it doubles as a GitHub repository. The first step is to set up a project website like this one by following the instructions below. 

## How does this website work?

First, check out the Github repository for this site: [https://github.com/alexAOS111204/alexAOS111204.github.io/](https://github.com/alexAOS111204/alexAOS111204.github.io/).

Using GitHub pages, you can write a website using markdown syntax - the same syntax we use to write comments in Google Colab notebooks. GitHub pages then takes the markdown file and renders it as a web page using a Jekyll theme. The markdown source code for this page [is shown here](https://github.com/alexAOS111204/alexAOS111204.github.io/blob/main/README.md?plain=1).

## Setting up your Project Website

### How to copy this site as a template
1. Create [a GitHub account](https://github.com/)
2.	Go to [https://github.com/alexAOS111204/alexAOS111204.github.io/](https://github.com/alexAOS111204/alexAOS111204.github.io/) and click *Use this template*, then **Create a new repository**. [![screenshot][1]][1]
3.	In the box that says *Repository name*, write your **Github username**, followed by **.github.io**, as shown in the screenshot below. Then click **Create repository** at the bottom. [![screenshot][2]][2]
4.	Go to the *Settings* tab, then click *Pages* (under *Code and automation*). In the *Build and deployment* section, under **Branch**, select "main" and click save (if it isn't already selected). It should look like this: [![screenshot][3]][3]
5.	Click the *Actions* tab at the top of the page and check that the build and deployment action has finished. Once it has, navigate to **[your username].github.io** to see your site, which should be a copy of this one! If you cannot see an *Actions* tab, just wait a few minutes then go to your URL to check it is live.

Now you are ready to customize your site! To add your name to the site, go to your repository page on Github, click `_config.yml`, and edit it to replace the temporary title with your name, etc. When we make changes to a project on Github, we have to **commit** the new version of each file. Github keeps track of all the changes we make, making it easy to roll back (i.e. return the project to a previous commit).

[1]: /assets/IMG/instr_new.png
[2]: /assets/IMG/instr_template.png
[3]: /assets/IMG/instr_bd.png

### How to change the theme (optional)
1.	You can choose any theme [listed on this page](https://github.com/pages-themes), be aware some do not work as well on mobile devices.
2.	From GitHub, edit `_config.yml` and replace the `theme:` line with `theme: jekyll-theme-name` where `name` is the name of the theme from the above repository (there are 13 to choose from, when I checked). **For example**, to use the `cayman` theme, use the line `theme: jekyll-theme-cayman`. You can check the *Actions* tab (as in step 5. above) to make sure the site is building successfully.

### How to change your site logo (optional)
1. Some themes, such as `jekyll-theme-minimal`, show a logo. In your repository, upload a logo or profile picture to the `assets/IMG/` directory
2. Open `_config.yml` and modify the line `logo: /assets/IMG/template_logo.png` to point to your new image

***

## Guide to Adding Content
* Your repository's `README.md` file (the file you are reading now) acts like a home page. Replace its contents with whatever you want the world to see by editing the file on GitHub.
* If you want to turn this page into a CV or blog, etc., it may be useful to refer to a [guide for writing Markdown](https://www.markdownguide.org/basic-syntax/).
* You can create other markdown files (.md) in your repository and navigate to them from this page using links, i.e.: [here is a link to another file, `project.md`](project.md)
* When editing a markdown file on GitHub, it is useful to wrap text by selecting the *Soft wrap* option as shown: ![screenshot](/assets/IMG/instr_wrap.png)
* If you want to get even more technical, you can also write HTML in your .md files, and GitHub Pages will render it. For example, the image below is displayed by writing the following (edit this file to see!): `<img align="right" width="200" height="200" src="/assets/IMG/template_frog.png">`
<img align="right" width="337" height="200" src="/assets/IMG/template_frog.png"> 

***

## Delivering your Project

Your final project is delivered in two components: a report and your code.

### Report

Your report should be **delivered via your website**. Submit a link to your website on BruinLearn so that your instructor can browse it to find your report. 

To make this simple, you can write the report using a word processor or Latex, then export it as a .pdf file and upload it to the `assets` directory. You can then link to it [like so](/assets/project_demo.pdf). However, you can also type the report directly onto the website using another markdown page - [here is](/project.md) a template for that.

### Code

A link to your code must be submitted on BruinLearn, and the course instructor must be able to download your code to mark it. The code could be in a Google Colab notebook (make sure to *share* the notebook so access is set to **Anyone with the link**), or you could upload the code into a separate GitHub repository, or you could upload the code into the `assets` directory of your website and link to it. 


<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## My Project

I applied machine learning techniques to investigate... Below is my report.

***

## Introduction 

Here is a summary description of the topic. Here is the problem. This is why the problem is important.

There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.

We did this to solve the problem. We concluded that...

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

<p>
When \(a \ne 0\), there are two solutions to \(ax^2 + bx + c = 0\) and they are
  \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
</p>

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)


