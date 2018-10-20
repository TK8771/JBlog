---
layout: post
title: "An Introduction to SQL"
date: 2018-10-20
excerpt: "A walkthrough of beginner-level SQL."
tags:
- SQL
- database management
- query
- data collection
image: "/assets/img/intro_to_sql.png"
---
![SQL]({{"/assets/img/intro_to_sql.png"}})

Structured Query Language (aka SQL) is a low-level programming language primarily used for managing and querying data in a structured database.

Specifically, it was built atop the research of Edgar Codd while he was at IBM's San Jose Research Laboratory. Despite this introduction, this post will not delve into the history of relational database management systems (aka RDBMS). However, Mr. Codd's work is the foundation of how database management systems work, and is the essential cornerstone of how *almost* all data is contained and stored in today's technology-driven world. I highly recommend doing some cursory reading on it here: ["Relational database management system"](https://en.wikipedia.org/wiki/Relational_database_management_system).

The main purpose of this post is to breakdown the basics of an SQL statement for learning purposes.

## Introduction to Key Concepts

In a relational database, we have individual tables that store data. Usually each table is responsible for holding some specific type on information: data on products, customers, sales, etc.

Let's say for example that we have the following tables and data in our database:

![Data]({{"/assets/img/sample_data.png"}})

Notice that each table has a column (named as tablename_id) that helps uniquely identify each row by a simple number count. The SALES table has sales_id, the CUSTOMER table has customer_id, etc. These are each referred to as the *primary key* of the table.

Now notice how each primary key also appears appears on other tables. For example, the SALES table also contains the product_id key and the customer_id key. When a primary key appears on another table, from that table's perspective, they are referred to as a *foreign key*. So the product_id and customer_id columns are foreign keys on the SALES table. These keys help us link tables to each other and extract valuable information in a relevant and customizable manner from the database.

To visualize how these tables can be 'linked' to each other:

![Tables]({{"/assets/img/sql_tables.jpg"}})

We'll see this 'linking' concept introduced more fully in the JOIN section.

## SELECT & FROM Clauses

The bread and butter of any SQL statement is the SELECT and FROM clauses, so let's say we wanted to select ALL data from the CUSTOMER table. We start with SELECT, then the star symbol (shift + 8) to say that we want all columns, and then FROM, and the table name itself. As such:

```SQL
SELECT *
FROM CUSTOMER
```
This would present us with the entire CUSTOMER table, exactly as you see it in the above image.

If we wanted to only select individual columns, we explicitly state those columns' names instead of the star symbol. Say we only wanted the customer_id, and the customer's name, it would look like this:

```SQL
SELECT customer_id, name
FROM CUSTOMER
```

## WHERE Clause

When dealing with large datasets, it will almost certainly be the case that you only want specific rows of your data, and not the whole table. In these cases, we use the WHERE clause to reduce the data returned.

Let's say we want to see all columns, but only for customer_id = 3, that would look like this:

```SQL
SELECT *
FROM CUSTOMER
WHERE customer_id = 3
```

This would give us this result:

![Where]({{"/assets/img/where.png"}})

## Explaining the JOIN clause

Beyond the basics, the JOIN clause is the most important concept to understand, and probably one of the most complicated when it comes to SQL. The JOIN clause allows us to link tables together and extract related data from separate tables.

Let's say we wanted to understand when customer_id = 3 bought his products. Notice that the date for sales appear on the SALES table, but the customer data is on the CUSTOMER table. We join the two where the primary key from one table equals the foreign key on another table. We may join the tables together in this manner:

```SQL
SELECT CUSTOMER.customer_id, CUSTOMER.name, SALES.data_of_sale
FROM CUSTOMER
  JOIN SALES on SALES.customer_id = CUSTOMER.customer_id
WHERE CUSTOMER.customer_id = 3
```
**Note: Notice that I put a TABLE_NAME.COLUMN_NAME before all of my variables.** Even though we're only required to explicitly state what table we're trying to pull from when there's overlap between column names on separate tables, it is generally best practice to always explicitly state the table you want the data from.

This would give us this result:

![JoinExample]({{"/assets/img/joins_exp.png"}})

This is the most basic version of a JOIN. There are several different types, depending on the data you're looking to include/exclude. Finally, I'll leave you with a diagram I still use regularly to this day to understand how different joins can work:

![Joins]({{"/assets/img/sql_joins.jpg"}})
