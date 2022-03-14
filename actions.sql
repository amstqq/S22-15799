create index review_i_id_idx on review (i_id);
create index item_i_id_idx on item (i_id);
create index review_i_id_u_id_idx on review (i_id,u_id);
create index trust_source_u_id_target_u_id_idx on trust (source_u_id,target_u_id);
create index useracct_u_id_idx on useracct (u_id);
create index review_u_id_rating_idx on review (u_id,rating);
create index review_i_id_rating_idx on review (i_id,rating);
