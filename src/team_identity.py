"""Read only user-supplied team identity for submission metadata."""
import json
from pathlib import Path


def load_identity(path='submission_team.json'):
    path=Path(path)
    if not path.exists(): return None
    data=json.loads(path.read_text(encoding='utf-8'))
    name=data.get('team_name','').strip()
    if not name or any(c in name for c in '<>:"/\\|?*') or any(ord(c)<32 for c in name):
        raise ValueError('Team name must be nonempty and valid in a Windows filename')
    members=data.get('members',[])
    if not isinstance(members,list) or any(not isinstance(m,str) or not m.strip() for m in members):
        raise ValueError('Members must be a list of supplied names')
    return dict(team_name=name,members=[m.strip() for m in members],affiliation=data.get('affiliation','').strip())


def methodology_identity(text,identity):
    if identity is None: return text
    lines=['**Team name:** '+identity['team_name'],
           '**Team members:** '+(', '.join(identity['members']) or 'Not supplied')]
    if identity['affiliation']: lines.append('**Affiliation:** '+identity['affiliation'])
    replacement='\n\n'.join(lines)
    if replacement in text: return text
    for placeholder in ('**Team name and members:** To be filled in by the submitting team.',
                        'Team name and members: to be filled in by the submitting team.'):
        if placeholder in text: return text.replace(placeholder,replacement,1)
    raise ValueError('Methodology identity placeholder not found')
