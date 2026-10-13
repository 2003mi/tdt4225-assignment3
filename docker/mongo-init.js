// Runs once, the first time the container starts with an empty data volume.
// Creates the app user with readWrite on DB_NAME (same as step 8 in the assignment).
const dbName = process.env.MONGO_INITDB_DATABASE;
const user = process.env.DB_USER;
const pwd = process.env.DB_PASSWORD;

db = db.getSiblingDB(dbName);
db.createUser({
  user: user,
  pwd: pwd,
  roles: [{ role: "readWrite", db: dbName }],
});

print(`Created user '${user}' with readWrite on '${dbName}'`);