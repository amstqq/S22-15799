CREATE INDEX idx_review_rating ON review(rating);
CREATE INDEX idx_review_i_id ON review(i_id);
CREATE INDEX idx_trust_source_u_id ON trust(source_u_id);
CREATE INDEX idx_review_u_id ON review(u_id);
CREATE INDEX idx_trust_target_u_id ON trust(target_u_id);
