import React from 'react';
import { FaBuilding, FaBuildingColumns } from 'react-icons/fa6';

// Academic affiliations get the columned-building glyph, everything else the
// plain office block. Keyed off the name so `affiliations` stays a plain list
// of strings in the template.
const ACADEMIC = /\b(universit|univ\.|college|school|institute|academy|laborator)/i;

const AffiliationIcon = ({ name }) => {
  const Icon = ACADEMIC.test(name) ? FaBuildingColumns : FaBuilding;
  return (
    <Icon
      size="0.9em"
      style={{ verticalAlign: '-0.1em', marginRight: '0.35em' }}
    />
  );
};

export default class Authors extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    if (!this.props.authors || !this.props.affiliations) {
      return null;
    }
    const columnMaxLen =
      this.props.authors.length > 4 ? 3 : this.props.authors.length;
    const authorClass = `uk-width-1-${columnMaxLen} uk-width-1-${this.props.authors.length}@m`;
    const affiliationClass = `uk-width-1-${this.props.affiliations.length} uk-margin-small-top`;
    return (
      <div>
        <div
          className="uk-text-primary uk-text-center uk-flex-center uk-grid-collapse"
          data-uk-grid
        >
          {this.props.authors.map((author, idx) => {
            // `mark` takes a single symbol or a list of them (e.g. equal
            // contribution *and* internship), rendered after the affiliation
            // numbers as `Name^{1,*,†}`.
            const marks = [].concat(author.mark ?? []);
            return (
              <span className={authorClass} key={'author-' + idx}>
                <a target="_blank" className="uk-link-toggle" href={author.url}>
                  {author.name}
                </a>
                <sup>{author.affiliation.concat(marks).join(',')}</sup>
              </span>
            );
          })}
        </div>
        <div
          className="uk-text-primary uk-text-center uk-grid-collapse"
          data-uk-grid
        >
          {this.props.affiliations.map((affiliation, idx) => {
            return (
              <span className={affiliationClass} key={'affiliation-' + idx}>
                <sup>{idx + 1}</sup>
                <AffiliationIcon name={affiliation} />
                {affiliation}
              </span>
            );
          })}
          {[].concat(this.props.meta ?? []).map((note, idx) => {
            // One footnote per row so multiple marks don't run together.
            return (
              <span className="uk-width-1-1" key={'meta-' + idx}>
                {note}
              </span>
            );
          })}
        </div>
      </div>
    );
  }
}
