```sql
CREATE TABLE Articles (
    ID_Article INTEGER PRIMARY KEY,     --
    Path TEXT NOT NULL,                 -- Topic path, e.g.: 'Newsroom/Sports/Motorsports/Formula 1'
    publishingDate TIMESTAMP NOT NULL,  --
    Title TEXT NOT NULL,                --
    Body TEXT                           -- Main article body, contains HTML markup
);

CREATE TABLE Posts (
    ID_Post INTEGER PRIMARY KEY,        --
    ID_Parent_Post INTEGER,             -- if this post is a reply: parent post's ID, otherwise NULL
    ID_Article INTEGER NOT NULL,        -- foreign key to 'Articles' table
    ID_User INTEGER NOT NULL,           --
    CreatedAt TIMESTAMP NOT NULL,       --
    Status TEXT,                        -- 'online' or 'deleted' (if deleted by moderator)
    Headline TEXT,                      -- Post headline (may be NULL if Body isn't)
    Body TEXT,                          -- Post main body (may be NULL if Headline isn't)
    PositiveVotes INTEGER NOT NULL,     -- Number of positive votes by other users
    NegativeVotes INTEGER NOT NULL      -- Number of negative votes by other users
);

-- This table lists all users who work for the newspaper (e.g. moderators, editorial journalists)
CREATE TABLE Newspaper_Staff (
    ID_User INTEGER PRIMARY KEY         -- matches with Posts.ID_User
);

-- This table may contain multiple annotator opinions for a given (ID_Post, Category) pair
CREATE TABLE Annotations (
    ID_Post INTEGER NOT NULL,           -- foreign key to 'Posts' table
    ID_Annotator INTEGER NOT NULL,      --
    Category TEXT NOT NULL,             -- name of the category, e.g. 'SentimentNegative'
    Value INTEGER NOT NULL,             -- 0 or 1, where 1 means the category does apply to the post
    PRIMARY KEY(ID_Post, ID_Annotator, Category)
);

-- This table will contain only one consolidated judgment for a given (ID_Post, Category) pair,
-- determined by a majority vote across all opinions in the 'Annotations' table
CREATE TABLE Annotations_consolidated (
    ID_Post INTEGER NOT NULL,           -- 
    Category TEXT NOT NULL,             -- name of the category, e.g. 'SentimentNegative'
    Value INTEGER NOT NULL,             -- 0 or 1, where 1 means the category does apply to the post
    PRIMARY KEY(ID_Post, Category)
);

-- This table is meant for reproducible cross validation. For each category, all posts are split
-- into ten folds in a stratified manner
-- https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29#k-fold_cross-validation
CREATE TABLE CrossValSplit(
    ID_Post INTEGER NOT NULL,           -- foreign key to 'Posts' table
    Category TEXT NOT NULL,             -- name of the category, e.g. 'SentimentNegative'
    Fold INTEGER NOT NULL,              -- from [1,10]
    PRIMARY KEY(ID_Post, Category, Fold)
);

-- This table defines the default ordering of the categories
CREATE TABLE Categories (
    Name TEXT PRIMARY KEY,              -- name of the category, e.g. 'SentimentNegative'
    Ord INTEGER                         -- ordering index
);
```
