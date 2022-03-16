create index observations_source_id_type_id_idx on observations (source_id,type_id);
create index observations_type_id_session_id_idx on observations (type_id,session_id);
