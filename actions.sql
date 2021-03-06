
drop index observations_source_id_idx;
drop index observations_session_id_idx;
drop index observations_created_time_idx;
drop index observations_type_id_idx;
create index observations_type_id_idx on observations (type_id);
create index observations_session_id_idx on observations (session_id);
create index observations_source_id_idx on observations (source_id);
create index observations_created_time_idx on observations (created_time);
