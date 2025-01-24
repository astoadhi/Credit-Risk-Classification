# Credit-Risk-Analysis
Credit risk is regarded as one of the most significant and substantial risks within the industry.
So this project aim is to predict the loan status (default or non-default) based on the given features related to the borrower and the loan itself.


# Database Objectives

1.	Design a relational database for a library management system. The application oversees multiple libraries, each hosting a diverse collection of books with varying quantities available for borrowing. Users can borrow or place holds on books (when the book is not immediately available for borrowing). 
2.	Populate the database with dummy data, to check if the database good or not. 
3.	Retrieve data to answer specific questions about the library's collection and circulation.


# Mission Statement 
The mission of this e-library application is to provide a user-friendly platform for managing multiple libraries, storing book information, and facilitating user registration, loan, and hold systems, while ensuring efficient tracking and management of book collections and user interactions.


# Business Rule

1. **Manage Multiple Libraries:**
    - The system should handle data for multiple libraries, each with its own collection of books.
2. **Book Collection:**
    - Store information about books, including titles, authors, genres, and copies available for borrowing.
    - Books should be categorized for easier searching (e.g., self-help, biography, fantasy, romance, historical).
3. **User Registration:**
    - Users can register on the e-library platform.
    - Registered users can interact with the platform by borrowing books, placing holds, and managing their account.
4. **Loan and Hold System:**
    - Users can borrow books from any library in this application if the book is available. 
    - The loan period is 2 weeks. Users can return books earlier than the due date
    - Books will be automatically returned when they exceed the due date
    - Users can only borrow 2 books at a time
    - The platform keeps track of loan transactions, including loan dates, due dates, and return dates.
    - Users can place holds on books that are currently unavailable.
    - The library maintains a hold queue, and when a book becomes available, it can be borrowed by the customer at the front of the queue. Additionally, if a customer doesn't borrow a held book within one week,  the book is released for other users to borrow. 
    - Users can only hold 2 books at the same time.


# Technical Requirements
* Database: PostgreSQL
* Programming Language: Python


# ERD Diagram
![](https://github.com/astoadhi/Database-for-eLibraries-Application/blob/main/ERD/ERD%20Diagram.png)

# Implementing the Database

After creating the ERD, implement the ERD results into the database using PostgreSQL and Data Definition Language (DDL). This [FILE](https://github.com/astoadhi/Database-for-eLibraries-Application/blob/main/postgreSQL/DDL.sql) is the syntax for making the database or you can go to [postgreSQL](postgreSQL) folder and click the DDL file.

# Generating Dummy Datasets

This project using python for the programming language. In generating dummy datasets, this project use libraries such as 'Faker','Pandas','datetime',and 'random'. You can use this [LINK](https://github.com/astoadhi/Database-for-eLibraries-Application/blob/main/Python/Generate%20Dummy%20Data.py) for see the script or you can go to [Python](Python) Folder and click the Generate Dummy Data.py

# Importing the Data

In this report, there are 8 csv files can be found in [dummy datasets](dummy_datasets) folder, they were generated by the script. You can adjust number of constant by yourself. The syntax for importing the data to postgreSQL in this [FILE](https://github.com/astoadhi/Database-for-eLibraries-Application/blob/main/postgreSQL/Importing_data.sql) (Note : adjust your file location, this syntax file is the writer file location)

# Question and Analysis

After construct the database, we must check the database, is it working properly or not. So in this project there are 5 SQL questions and analysis to have the insight how the database work. The SQL queries file  for answering the question can be found [HERE](https://github.com/astoadhi/Database-for-eLibraries-Application/blob/main/postgreSQL/Query%20Answers.sql).
1. Which libraries have the most books?
2. Which categories are most popular among users?
3. Which users have borrowed the most books?
4. Which authors have the most borrowed books?
5. Which book categories have the longest average wait times?
