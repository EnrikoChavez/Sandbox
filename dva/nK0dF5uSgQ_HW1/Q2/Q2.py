########################### DO NOT MODIFY THIS SECTION ##########################
#################################################################################
import sqlite3
from sqlite3 import Error
import csv
#################################################################################

## Change to False to disable Sample
SHOW = False

############### SAMPLE CLASS AND SQL QUERY ###########################
######################################################################
class Sample():
    def sample(self):
        
        try:
            connection = sqlite3.connect("sample")
            connection.text_factory = str
        except Error as e:
            print("Error occurred: " + str(e))
        print('\033[32m' + "Sample: " + '\033[m')
        
        # Sample Drop table
        connection.execute("DROP TABLE IF EXISTS sample;")
        # Sample Create
        connection.execute("CREATE TABLE sample(id integer, name text);")
        # Sample Insert
        connection.execute("INSERT INTO sample VALUES (?,?)",("1","test_name"))
        connection.commit()
        # Sample Select
        cursor = connection.execute("SELECT * FROM sample;")
        print(cursor.fetchall())

######################################################################

class HW2_sql():
    ############### DO NOT MODIFY THIS SECTION ###########################
    ######################################################################
    def create_connection(self, path):
        connection = None
        try:
            connection = sqlite3.connect(path)
            connection.text_factory = str
        except Error as e:
            print("Error occurred: " + str(e))
    
        return connection

    def execute_query(self, connection, query):
        cursor = connection.cursor()
        try:
            if query == "":
                return "Query Blank"
            else:
                cursor.execute(query)
                connection.commit()
                return "Query executed successfully"
        except Error as e:
            return "Error occurred: " + str(e)
    ######################################################################
    ######################################################################

    # GTusername [0 points]
    def GTusername(self):
        gt_username = "echavez33"
        return gt_username
    
    # Part a.i Create Tables [2 points]
    def part_ai_1(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_ai_1_sql = "CREATE TABLE movies(id integer, title text, score real);"
        ######################################################################
        
        return self.execute_query(connection, part_ai_1_sql)

    def part_ai_2(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_ai_2_sql = "CREATE TABLE movie_cast(movie_id integer, cast_id integer, cast_name text, birthday text, popularity real);"
        ######################################################################
        
        return self.execute_query(connection, part_ai_2_sql)
    
    # Part a.ii Import Data [2 points]
    def part_aii_1(self,connection,path):
        ############### CREATE IMPORT CODE BELOW ############################
        with open(path, 'r') as file:
            reader = csv.DictReader(file, fieldnames=["id", "title", "score"])
            for row in reader:
                row["id"] = int(row["id"])
                row["score"] = float(row["score"].strip())
                connection.execute("INSERT INTO movies (id, title, score) VALUES (?,?,?)", [row["id"], row["title"], row["score"]])
        connection.commit()
       ######################################################################
        
        sql = "SELECT COUNT(id) FROM movies;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]
    
    def part_aii_2(self,connection, path):
        ############### CREATE IMPORT CODE BELOW ############################
        with open(path, 'r') as file:
            reader = csv.DictReader(file, fieldnames=["movie_id", "cast_id", "cast_name", "birthday", "popularity"])
            for row in reader:
                row["movie_id"] = int(row["movie_id"])
                row["cast_id"] = int(row["cast_id"])
                row["popularity"] = float(row["popularity"].strip())
                connection.execute("INSERT INTO movie_cast (movie_id, cast_id, cast_name, birthday, popularity) VALUES (?,?,?,?,?)",
                                    [row["movie_id"], row["cast_id"], row["cast_name"], row["birthday"], row["popularity"]])
        connection.commit()
        ######################################################################
        
        sql = "SELECT COUNT(cast_id) FROM movie_cast;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]

    # Part a.iii Vertical Database Partitioning [5 points]
    def part_aiii(self,connection):
        ############### EDIT CREATE TABLE SQL STATEMENT ###################################
        part_aiii_sql = "CREATE TABLE cast_bio(cast_id integer, cast_name text, birthday text, popularity real);"
        ######################################################################
        
        self.execute_query(connection, part_aiii_sql)
        
        ############### CREATE IMPORT CODE BELOW ############################
        part_aiii_insert_sql = "INSERT INTO cast_bio(cast_id, cast_name, birthday, popularity) SELECT DISTINCT cast_id, cast_name, birthday, popularity FROM movie_cast;"
        ######################################################################
        
        self.execute_query(connection, part_aiii_insert_sql)
        
        sql = "SELECT COUNT(cast_id) FROM cast_bio;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]


    # Part b Create Indexes [1 points]
    def part_b_1(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_b_1_sql = "CREATE INDEX movie_index ON movies(id);"
        ######################################################################
        return self.execute_query(connection, part_b_1_sql)
    
    def part_b_2(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_b_2_sql = "CREATE INDEX cast_index ON movie_cast(cast_id);"
        ######################################################################
        return self.execute_query(connection, part_b_2_sql)
    
    def part_b_3(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_b_3_sql = "CREATE INDEX cast_bio_index ON cast_bio(cast_id);"
        ######################################################################
        return self.execute_query(connection, part_b_3_sql)
    
    # Part c Calculate a Proportion [3 points]
    def part_c(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_c_sql = "SELECT printf('%.2f', CAST(COUNT(*) as float) * 100 /(SELECT COUNT(*) FROM movies)) FROM movies WHERE score >= 7 AND score <= 20;"
        ######################################################################
        cursor = connection.execute(part_c_sql)
        return cursor.fetchall()[0][0]

    # Part d Find the Most Prolific Actors [4 points]
    def part_d(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_d_sql = """SELECT cast_name, count(movie_id) AS appearance_count 
        FROM movie_cast 
        WHERE popularity > 10 
        GROUP BY cast_id 
        ORDER BY appearance_count DESC, cast_name ASC 
        LIMIT 5"""
        ######################################################################
        cursor = connection.execute(part_d_sql)
        return cursor.fetchall()

    # Part e Find the Highest Scoring Movies With the Least Amount of Cast [4 points]
    def part_e(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_e_sql = """SELECT title, printf('%.2f',score), count(cast_id) AS cast_size 
        FROM movies 
        INNER JOIN movie_cast ON movies.id = movie_cast.movie_id 
        GROUP BY movie_id 
        ORDER BY score DESC, cast_size ASC, title ASC 
        LIMIT 5"""
        ######################################################################
        cursor = connection.execute(part_e_sql)
        return cursor.fetchall()
    
    # Part f Get High Scoring Actors [4 points]
    def part_f(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_f_sql = """
        WITH movies_f AS
        (SELECT * FROM movies WHERE score >= 25),
        movie_cast_ff AS
        (SELECT * FROM movie_cast as mv
                    INNER JOIN movies
                    ON movies.id = mv.movie_id
                    WHERE score >= 25),
        app_c AS
        (SELECT cast_id, count(movie_cast_ff.movie_id) AS appearance_count
                FROM movie_cast_ff
                GROUP BY movie_cast_ff.cast_id),
        movie_cast_f AS
        (SELECT * FROM app_c
                INNER JOIN movie_cast
                ON movie_cast.cast_id = app_c.cast_id)
        SELECT movie_cast_f.cast_id, movie_cast_f.cast_name, printf('%.2f',avg(score)) AS avg_score 
        FROM movies_f
        INNER JOIN movie_cast_f 
        ON movies_f.id = movie_cast_f.movie_id
        WHERE movie_cast_f.appearance_count >= 3
        GROUP BY movie_cast_f.cast_id
        ORDER BY avg_score DESC, movie_cast_f.cast_name ASC
        LIMIT 10"""
        ######################################################################
        cursor = connection.execute(part_f_sql)
        return cursor.fetchall()

    # Part g Creating Views [6 points]
    def part_g(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_g_sql = """
        CREATE VIEW good_collaboration AS
        WITH pairs AS
        (SELECT min(m1.cast_id,m2.cast_id) AS min_,
                max(m1.cast_id,m2.cast_id) AS max_
        FROM cast_bio AS m1
        JOIN cast_bio AS m2
        ON m1.cast_id <> m2.cast_id
        GROUP by 1, 2)
        SELECT *
        FROM pairs
        WHERE
           (SELECT count(*) as movies_tog
            FROM movie_cast as mc1
            JOIN movie_cast as mc2
            ON mc1.movie_id = mc2.movie_id
            WHERE mc1.cast_id = min_
            AND mc2.cast_id = max_) >= 2
        """
        ######################################################################
        return self.execute_query(connection, part_g_sql)
    
    def part_gi(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_g_i_sql = ""
        ######################################################################
        cursor = connection.execute(part_g_i_sql)
        return cursor.fetchall()
    
    # Part h FTS [4 points]
    def part_h(self,connection,path):
        ############### EDIT SQL STATEMENT ###################################
        part_h_sql = ""
        ######################################################################
        connection.execute(part_h_sql)
        ############### CREATE IMPORT CODE BELOW ############################
        
        ######################################################################
        sql = "SELECT COUNT(id) FROM movie_overview;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]
        
    def part_hi(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_hi_sql = ""
        ######################################################################
        cursor = connection.execute(part_hi_sql)
        return cursor.fetchall()[0][0]
    
    def part_hii(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_hii_sql = ""
        ######################################################################
        cursor = connection.execute(part_hii_sql)
        return cursor.fetchall()[0][0]


if __name__ == "__main__":
    
    ########################### DO NOT MODIFY THIS SECTION ##########################
    #################################################################################
    if SHOW == True:
        sample = Sample()
        sample.sample()

    print('\033[32m' + "Q2 Output: " + '\033[m')
    db = HW2_sql()
    try:
        conn = db.create_connection("Q2")
    except:
        print("Database Creation Error")

    try:
        conn.execute("DROP TABLE IF EXISTS movies;")
        conn.execute("DROP TABLE IF EXISTS movie_cast;")
        conn.execute("DROP TABLE IF EXISTS cast_bio;")
        conn.execute("DROP VIEW IF EXISTS good_collaboration;")
        conn.execute("DROP TABLE IF EXISTS movie_overview;")
    except Exception as e:
        print("Error in Table Drops")
        print(e)

    try:
        print('\033[32m' + "part ai 1: " + '\033[m' + str(db.part_ai_1(conn)))
        print('\033[32m' + "part ai 2: " + '\033[m' + str(db.part_ai_2(conn)))
    except Exception as e:
         print("Error in Part a.i")
         print(e)

    try:
        print('\033[32m' + "Row count for Movies Table: " + '\033[m' + str(db.part_aii_1(conn,"data/movies.csv")))
        print('\033[32m' + "Row count for Movie Cast Table: " + '\033[m' + str(db.part_aii_2(conn,"data/movie_cast.csv")))
    except Exception as e:
        print("Error in part a.ii")
        print(e)

    try:
        print('\033[32m' + "Row count for Cast Bio Table: " + '\033[m' + str(db.part_aiii(conn)))
    except Exception as e:
        print("Error in part a.iii")
        print(e)

    try:
        print('\033[32m' + "part b 1: " + '\033[m' + db.part_b_1(conn))
        print('\033[32m' + "part b 2: " + '\033[m' + db.part_b_2(conn))
        print('\033[32m' + "part b 3: " + '\033[m' + db.part_b_3(conn))
    except Exception as e:
        print("Error in part b")
        print(e)

    try:
        print('\033[32m' + "part c: " + '\033[m' + str(db.part_c(conn)))
    except Exception as e:
        print("Error in part c")
        print(e)

    try:
        print('\033[32m' + "part d: " + '\033[m')
        for line in db.part_d(conn):
            print(line[0],line[1])
    except Exception as e:
        print("Error in part d")
        print(e)

    try:
        print('\033[32m' + "part e: " + '\033[m')
        for line in db.part_e(conn):
            print(line[0],line[1],line[2])
    except Exception as e:
        print("Error in part e")
        print(e)

    try:
        print('\033[32m' + "part f: " + '\033[m')
        for line in db.part_f(conn):
            print(line[0],line[1],line[2])
    except Exception as e:
        print("Error in part f")
        print(e)
    
    try:
        print('\033[32m' + "part g: " + '\033[m' + str(db.part_g(conn)))
        # print("\033[32mRow count for good_collaboration view:\033[m", conn.execute("select * from good_collaboration").fetchall())
        print("\033[32mRow count for good_collaboration view:\033[m", conn.execute("select count(*) from good_collaboration").fetchall()[0][0])
        print('\033[32m' + "part g.i: " + '\033[m')
        for line in db.part_gi(conn):
            print(line[0],line[1],line[2])
    except Exception as e:
        print("Error in part g")
        print(e)

    try:   
        print('\033[32m' + "part h: " + '\033[m'+ str(db.part_h(conn,"data/movie_overview.csv")))
        print('\033[32m' + "Count h.i: " + '\033[m' + str(db.part_hi(conn)))
        print('\033[32m' + "Count h.ii: " + '\033[m' + str(db.part_hii(conn)))
    except Exception as e:
        print("Error in part h")
        print(e)

    conn.close()
    #################################################################################
    #################################################################################
  
