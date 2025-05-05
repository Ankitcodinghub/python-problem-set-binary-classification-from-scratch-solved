# python-problem-set-binary-classification-from-scratch-solved
**TO GET THIS SOLUTION VISIT:** [Python Problem Set-Binary Classification from Scratch Solved](https://www.ankitcodinghub.com/product/python-problem-set-binary-classification-from-scratch-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;99657&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Python Problem Set-Binary Classification from Scratch Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
# Binary Classification from Scratch

Problem statement:

Consider the binary classification problem of mapping a given input to two classes. Let X = Rd and Y =-1,+1 be the input space and output space, respectively. In simple words, it means that the input has d features and all of them are real valued, whereas the output can only take values -1,+1. This is one of the most common problems in machine learning and many sophisticated methods exist to solve it.

• In the question, we will solve it using the concepts we have already learned in class. Let us assume the two sets of points can be separated using a straight line i.e. the samples are linearly separable. So if d = 2, one should be able to draw a line to distinguish between the two classes. All points lying on side of the line should belong to a particular class (say 1) and the points lying on the other side should belong to another class (say 2). To see what this would look like, your first task is as follows:

Write a function that will randomly generate a dataset for this problem. Your function should randomly choose a line l, which can be denoted as ax + by + c = 0 and the line divides the plane into two sides. On one side, ax + by + c &gt; 0 while on the other ax + by + c &lt; 0. Use this fact to randomly generate k0 points on the side of class 0 (i.e. y = −1 ) and k1 points on the side of class 1 (i.e. y = 1). Create a plot of this dataset where all the points corresponding to one class are blue and those of the other class are green, the line dividing both classes should be red. Axes should be labeled.Note: Do not confuse the x and y in the equation of line ax + by + c = 0 with X and Y . Instead imagine these x and y as the 2D coordinate system on which you have different points which should lie on 2 sides of the line ax + by + c = 0. For example, there is a point (2, 3) in the 2D system where x = 2 and y = 3.

• If Y is the variable you are trying to predict using a feature X then in a typical Machine Learning

problem, you are tasked with a target function f which maps X to Y i.e. Find f such that Y = f(X).

When you are given a dataset for which you do not have access the target function f, you have to

learn it from the data. In this problem, we are going to learn the parameters of the line that separates

the two classes for the dataset that we constructed in problem above. As we previously mentioned,

that line can be represented as ax + by + c = 0.

The goal here is to correctly find out the coefficients a, b, and c, represented below as w which is a

vector. The algorithm to find it is a simple iterative process:

– Randomly choose a w to begin with.

– Keep on adjusting the value of w as follows until all data samples are correctly classified:

∗ Randomly choose a sample from the dataset without replacement and see if it is correctly

classified. If yes, move on to another sample.

∗ If not, then update the weights as wt+1 = wt + yx and go back to the previous step (of

randomly choosing a sample)

· wt+1 is value of w at iteration t + 1

· wt+1 is value of w at iteration t

· y is the class label for the sample under consideration

· x is the data-point under consideration

Write a function that implements this learning algorithm. The input to the function is going to be a

dataset represented by the input variable X and the target variable Y . The output of the function

should be the chosen w.
