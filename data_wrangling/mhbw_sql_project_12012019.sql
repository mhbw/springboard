
-- two stylistic notes
	-- 1) the GUI adds tic marks around certain text when you auto-fill. In some spots I left them in others I took them out. If I was writing this on command line I'd take them out, but I left them in this case.
	-- 2) similarly the GUI didn't require a schema designation. Again, I wouldn't go without if this was outside the GUI but it functions here. 

/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */
SELECT DISTINCT name
FROM  Facilities 
WHERE membercost > 0.0;

/*Tennis Court 1
Tennis Court 2
Massage Room 1
Massage Room 2
Squash Court
*/

/* Q2: How many facilities do not charge a fee to members? */

SELECT COUNT( DISTINCT facid ) 
FROM  Facilities 
WHERE membercost = 0.0;

-- 4 facilities

/*Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

SELECT  name 
, Facilities.facid
,  membercost
,  monthlymaintenance 
, per_main_covered-- this wasn't asked for but I felt like it was interesting
FROM  
	Facilities 
INNER JOIN (
	SELECT facid
		, membercost / monthlymaintenance AS per_main_covered
	FROM  
		Facilities 
	WHERE 
		membercost >0
			)has_fee 
ON 
	has_fee.facid = Facilities.facid;


/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */

SELECT * 
FROM  Facilities 
WHERE facid
IN ( 1, 5 ) ;


/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */

SELECT name
	, monthlymaintenance
	, CASE WHEN monthlymaintenance > 100 THEN 'expensive' ELSE 'cheap' END as valuation
FROM Facilities;



/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */

SELECT firstname, surname
FROM Members
INNER JOIN (

SELECT MAX( joindate ) AS joindate
FROM Members
) AS jd ON jd.joindate = Members.joindate;

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

SELECT DISTINCT CONCAT(Members.firstname,' ',Members.surname) as member_name 
	, CASE WHEN Bookings.facid=0 THEN 'Tennis Court 1' ELSE 'Tennis Court 2' END as facilityname
FROM  
	Members
LEFT JOIN
	Bookings
ON
	Bookings.memid = Members.memid
WHERE
	facid 
IN
	(0,1)
ORDER BY 1 ASC;

 /*Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

SELECT DISTINCT Facilities.name
, Bookings.bookid
, CONCAT(Members.firstname,' ',Members.surname) as member_name 
, CASE WHEN Members.memid=0 THEN slots*guestcost ELSE slots*membercost END AS cost
FROM  
	Members
LEFT JOIN
	Bookings
ON
	Bookings.memid = Members.memid
LEFT JOIN
	Facilities
ON
	Facilities.facid = Bookings.facid
WHERE
	LEFT(Bookings.starttime, 10)='2012-09-14'
HAVING cost > 30
ORDER BY 4 desc;

/* Q9: This time, produce the same result as in Q8, but using a subquery. */

SELECT DISTINCT NewBookings.name, NewBookings.bookid, CONCAT( Members.firstname,  ' ', Members.surname ) AS member_name, cost
FROM 
	Members
LEFT JOIN 
	(
	SELECT DISTINCT memid
		, bookid
		, Facilities.name 
		, CASE WHEN Bookings.memid =0 THEN slots * guestcost ELSE slots * membercost END AS cost
	FROM 
		Bookings
	LEFT JOIN 
		Facilities 
	ON 
		Bookings.facid = Facilities.facid
	WHERE 
		LEFT( Bookings.starttime, 10 ) =  '2012-09-14'
	) 
	NewBookings 
ON 
	NewBookings.memid = Members.memid
WHERE 
	cost >30
ORDER BY 4 DESC ;

/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

select distinct revenue.name
	, sum(cost) as total_income
FROM
(
SELECT DISTINCT memid
		, bookid
		, Facilities.name 
		, CASE WHEN Bookings.memid =0 THEN slots * guestcost ELSE slots * membercost END AS cost
	FROM 
		Bookings
	LEFT JOIN 
		Facilities 
	ON 
		Bookings.facid = Facilities.facid
) revenue
group by 1
HAVING total_income < 1000
 order by 2 DESC;


