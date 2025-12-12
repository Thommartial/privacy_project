#!/usr/bin/env python3
"""
Generate realistic documents with diverse PII types for testing.
"""

import os
import random
import json
from pathlib import Path
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Tuple
import re

class PIIDocumentGenerator:
    def __init__(self, output_dir: str = "pii_documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data pools
        self._init_name_pools()
        self._init_geo_pools()
        self._init_contact_pools()
        self._init_financial_pools()
        self._init_digital_pools()
        self._init_demographic_pools()
        
    def _init_name_pools(self):
        """Initialize pools for names and personal identifiers."""
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", 
            "Michael", "Linda", "William", "Elizabeth", "David", "Barbara",
            "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
            "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
            "Matthew", "Margaret", "Anthony", "Betty", "Donald", "Sandra"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
            "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
            "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris"
        ]
        
        self.prefixes = ["Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Rev."]
        self.suffixes = ["Jr.", "Sr.", "II", "III", "PhD", "MD", "DDS"]
        
    def _init_geo_pools(self):
        """Initialize pools for geographic data."""
        self.cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
            "San Francisco", "Indianapolis", "Seattle", "Denver", "Washington"
        ]
        
        self.states = [
            "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
            "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"
        ]
        
        self.streets = [
            "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine Rd",
            "Elm St", "Washington Ave", "Park Blvd", "Lake St", "Hill Rd",
            "Spring St", "River Dr", "Sunset Blvd", "Broadway", "Market St"
        ]
        
        self.counties = [
            "Los Angeles County", "Cook County", "Harris County", "Maricopa County",
            "San Diego County", "Orange County", "Miami-Dade County", "Dallas County",
            "King County", "Santa Clara County"
        ]
        
    def _init_contact_pools(self):
        """Initialize pools for contact information."""
        self.domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
            "company.com", "business.org", "university.edu", "hospital.org"
        ]
        
        self.area_codes = ["212", "310", "312", "415", "617", "713", "818", "917"]
        
    def _init_financial_pools(self):
        """Initialize pools for financial data."""
        self.banks = [
            "Chase", "Bank of America", "Wells Fargo", "Citibank", "US Bank",
            "Capital One", "PNC", "TD Bank", "HSBC", "Barclays"
        ]
        
    def _init_digital_pools(self):
        """Initialize pools for digital identifiers."""
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
            "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/68.0"
        ]
        
        self.devices = [
            "iPhone 12", "Samsung Galaxy S21", "Google Pixel 5", "Dell XPS 13",
            "MacBook Pro", "iPad Pro", "Surface Pro 7", "ThinkPad X1"
        ]
        
    def _init_demographic_pools(self):
        """Initialize pools for demographic information."""
        self.job_titles = [
            "Software Engineer", "Data Scientist", "Product Manager", "Marketing Director",
            "Sales Manager", "HR Specialist", "Financial Analyst", "Operations Manager",
            "Consultant", "Research Scientist", "Professor", "Doctor", "Lawyer",
            "Architect", "Designer", "Accountant", "Project Manager", "Business Analyst"
        ]
        
        self.departments = [
            "Engineering", "Marketing", "Sales", "Human Resources", "Finance",
            "Operations", "Research & Development", "Legal", "IT", "Customer Support"
        ]
        
        self.genders = ["Male", "Female", "Non-binary"]
        
    def generate_random_name(self, include_prefix=False, include_suffix=False):
        """Generate a random full name."""
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        middle = random.choice(self.first_names) if random.random() > 0.7 else None
        
        parts = []
        if include_prefix and random.random() > 0.5:
            parts.append(random.choice(self.prefixes))
        
        parts.append(first)
        
        if middle:
            parts.append(middle)
            parts.append(last)
        else:
            parts.append(last)
            
        if include_suffix and random.random() > 0.5:
            parts.append(random.choice(self.suffixes))
            
        return " ".join(parts)
    
    def generate_email(self, first_name=None, last_name=None):
        """Generate a random email address."""
        if first_name is None:
            first_name = random.choice(self.first_names).lower()
        if last_name is None:
            last_name = random.choice(self.last_names).lower()
            
        patterns = [
            f"{first_name}.{last_name}@{random.choice(self.domains)}",
            f"{first_name[0]}{last_name}@{random.choice(self.domains)}",
            f"{first_name}_{last_name}{random.randint(10, 99)}@{random.choice(self.domains)}",
            f"{first_name}{random.randint(1970, 2000)}@{random.choice(self.domains)}"
        ]
        
        return random.choice(patterns)
    
    def generate_phone(self):
        """Generate a random phone number."""
        area = random.choice(self.area_codes)
        prefix = f"{random.randint(200, 999)}"
        line = f"{random.randint(1000, 9999)}"
        
        formats = [
            f"({area}) {prefix}-{line}",
            f"{area}-{prefix}-{line}",
            f"+1 {area} {prefix} {line}",
            f"{area}.{prefix}.{line}"
        ]
        
        return random.choice(formats)
    
    def generate_address(self):
        """Generate a random address."""
        building = random.randint(100, 9999)
        street = random.choice(self.streets)
        city = random.choice(self.cities)
        state = random.choice(self.states)
        zip_code = f"{random.randint(10000, 99999)}"
        
        return {
            "building": building,
            "street": street,
            "city": city,
            "state": state,
            "zip": zip_code
        }
    
    def generate_ssn(self):
        """Generate a random Social Security Number."""
        area = f"{random.randint(100, 999)}"
        group = f"{random.randint(10, 99)}"
        serial = f"{random.randint(1000, 9999)}"
        
        formats = [
            f"{area}-{group}-{serial}",
            f"{area} {group} {serial}",
            f"{area}{group}{serial}"
        ]
        
        return random.choice(formats)
    
    def generate_credit_card(self):
        """Generate a random credit card number."""
        # Different card types have different starting digits
        card_types = [
            ("4", 16),  # Visa
            ("5", 16),  # MasterCard
            ("3", 15),  # American Express
            ("6", 16),  # Discover
        ]
        
        start, length = random.choice(card_types)
        # Generate remaining digits
        remaining = "".join([str(random.randint(0, 9)) for _ in range(length - 1)])
        return f"{start}{remaining}"
    
    def generate_ip_address(self, version=4):
        """Generate a random IP address."""
        if version == 4:
            return ".".join(str(random.randint(1, 255)) for _ in range(4))
        else:  # IPv6
            return ":".join(f"{random.randint(0x1000, 0xffff):x}" for _ in range(8))
    
    def generate_mac_address(self):
        """Generate a random MAC address."""
        return ":".join(f"{random.randint(0x00, 0xff):02x}" for _ in range(6))
    
    def generate_url(self):
        """Generate a random URL."""
        domains = ["example.com", "test.org", "demo.net", "sample.edu", "website.io"]
        subdomains = ["www", "app", "api", "blog", "store", "support"]
        protocols = ["https://", "http://"]
        
        protocol = random.choice(protocols)
        subdomain = random.choice(subdomains) if random.random() > 0.3 else ""
        domain = random.choice(domains)
        path = f"/{random.choice(['page', 'article', 'post', 'product'])}/{random.randint(1, 100)}" if random.random() > 0.5 else ""
        
        url = f"{protocol}"
        if subdomain:
            url += f"{subdomain}."
        url += f"{domain}{path}"
        
        return url
    
    def generate_bitcoin_address(self):
        """Generate a fake Bitcoin address."""
        chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        return "1" + "".join(random.choice(chars) for _ in range(33))
    
    def generate_ethereum_address(self):
        """Generate a fake Ethereum address."""
        chars = "0123456789abcdef"
        return "0x" + "".join(random.choice(chars) for _ in range(40))
    
    def generate_iban(self):
        """Generate a fake IBAN."""
        country = random.choice(["DE", "GB", "FR", "IT", "ES", "NL"])
        check = f"{random.randint(10, 99)}"
        bank = f"{random.randint(10000000, 99999999)}"
        account = f"{random.randint(1000000000, 9999999999)}"
        return f"{country}{check} {bank} {account}"
    
    def generate_bic(self):
        """Generate a fake BIC/SWIFT code."""
        bank = random.choice(["DEUT", "COBA", "HSBC", "BARB", "CITI"])
        country = random.choice(["DE", "GB", "FR", "US", "CH"])
        location = random.choice(["FF", "MM", "LX", "NY", "LN"])
        branch = random.choice(["XXX", "ABC", "DEF", "GHI"])
        return f"{bank}{country}{location}{branch}"
    
    def generate_account_number(self):
        """Generate a random account number."""
        length = random.choice([8, 10, 12])
        return "".join(str(random.randint(0, 9)) for _ in range(length))
    
    def generate_gps_coordinates(self):
        """Generate random GPS coordinates."""
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        return f"{lat:.6f}, {lon:.6f}"
    
    def generate_date(self, start_year=2010, end_year=2024):
        """Generate a random date."""
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Safe day for all months
        return f"{year}-{month:02d}-{day:02d}"
    
    def generate_document_template(self, doc_type):
        """Generate a document template based on type."""
        templates = {
            "email": [
                "Subject: {subject}\n\nDear {fullname},\n\n{body}\n\nBest regards,\n{sender}",
                "From: {email_from}\nTo: {email_to}\nSubject: {subject}\n\n{body}",
            ],
            "report": [
                "Report for {company}\nDate: {date}\nAuthor: {author}\n\n{body}",
                "{report_type} Analysis\n\nExecutive Summary:\n{summary}\n\nDetailed Findings:\n{findings}",
            ],
            "letter": [
                "{company_address}\n\n{date}\n\nDear {recipient},\n\n{body}\n\nSincerely,\n{sender}",
            ],
            "meeting": [
                "Meeting Minutes\nDate: {date}\nTime: {time}\nLocation: {location}\nAttendees: {attendees}\n\nAgenda:\n{agenda}\n\nAction Items:\n{actions}",
            ],
            "medical": [
                "Patient: {patient_name}\nDOB: {dob}\nMRN: {mrn}\nDate: {date}\n\nChief Complaint: {complaint}\n\nAssessment: {assessment}\n\nPlan: {plan}",
            ],
            "financial": [
                "Statement for Account: {account_number}\nPeriod: {period}\n\nTransactions:\n{transactions}\n\nBalance: {balance}",
            ],
            "legal": [
                "Case Number: {case_number}\nClient: {client_name}\nDate: {date}\n\n{body}",
            ],
            "support": [
                "Ticket #{ticket_number}\nCustomer: {customer_name}\nEmail: {customer_email}\nIssue: {issue}\n\nResolution: {resolution}",
            ]
        }
        
        return random.choice(templates.get(doc_type, templates["email"]))
    
    def generate_document(self, doc_id):
        """Generate a single document with various PII types."""
        doc_type = random.choice(["email", "report", "letter", "meeting", "medical", "financial", "legal", "support"])
        template = self.generate_document_template(doc_type)
        
        # Generate various PII elements
        fullname1 = self.generate_random_name(include_prefix=True)
        fullname2 = self.generate_random_name(include_prefix=True)
        company_name = f"{random.choice(self.last_names)} & {random.choice(self.last_names)}"
        
        email1 = self.generate_email()
        email2 = self.generate_email()
        
        phone1 = self.generate_phone()
        phone2 = self.generate_phone()
        
        address1 = self.generate_address()
        address2 = self.generate_address()
        
        ssn = self.generate_ssn()
        credit_card = self.generate_credit_card()
        bitcoin = self.generate_bitcoin_address()
        ethereum = self.generate_ethereum_address()
        iban = self.generate_iban()
        bic = self.generate_bic()
        account_num = self.generate_account_number()
        
        ipv4 = self.generate_ip_address(4)
        ipv6 = self.generate_ip_address(6)
        mac = self.generate_mac_address()
        url = self.generate_url()
        
        gps = self.generate_gps_coordinates()
        date = self.generate_date()
        
        job_title = random.choice(self.job_titles)
        department = random.choice(self.departments)
        
        # Fill template with data
        doc_data = {
            "subject": f"Regarding your {random.choice(['account', 'order', 'application', 'request'])} #{random.randint(1000, 9999)}",
            "fullname": fullname1,
            "body": self.generate_body_content(),
            "sender": fullname2,
            "email_from": email1,
            "email_to": email2,
            "company": company_name,
            "date": date,
            "author": fullname1,
            "report_type": random.choice(["Quarterly", "Annual", "Performance", "Financial"]),
            "summary": f"This report summarizes {random.choice(['Q3 performance', 'annual results', 'market analysis'])}.",
            "findings": f"Key finding: {random.choice(['Revenue increased by 15%', 'Customer satisfaction improved', 'Costs were reduced'])}.",
            "company_address": f"{address1['building']} {address1['street']}, {address1['city']}, {address1['state']} {address1['zip']}",
            "recipient": fullname2,
            "time": f"{random.randint(9, 16)}:{random.choice(['00', '15', '30', '45'])}",
            "location": f"{random.choice(['Conference Room A', 'Zoom Meeting', 'Office'])}",
            "attendees": f"{fullname1}, {fullname2}, {self.generate_random_name()}",
            "agenda": f"1. {random.choice(['Review Q3 results', 'Discuss new strategy', 'Budget planning'])}\n2. {random.choice(['Team updates', 'Project review', 'Next steps'])}",
            "actions": f"- {fullname1}: {random.choice(['Prepare report', 'Follow up with client', 'Review documents'])}\n- {fullname2}: {random.choice(['Schedule meeting', 'Update website', 'Contact vendor'])}",
            "patient_name": fullname1,
            "dob": self.generate_date(1950, 2000),
            "mrn": f"MRN{random.randint(100000, 999999)}",
            "complaint": random.choice(["Headache", "Fever", "Back pain", "Fatigue"]),
            "assessment": random.choice(["Viral infection", "Muscle strain", "Allergic reaction"]),
            "plan": random.choice(["Prescribe medication", "Follow up in 2 weeks", "Refer to specialist"]),
            "account_number": account_num,
            "period": f"{random.choice(['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec'])} {random.randint(2020, 2024)}",
            "transactions": f"{date}: {random.choice(['Payment received', 'Withdrawal', 'Transfer'])} ${random.uniform(10, 1000):.2f}",
            "balance": f"${random.uniform(1000, 10000):.2f}",
            "case_number": f"CASE-{random.randint(10000, 99999)}",
            "client_name": fullname1,
            "ticket_number": random.randint(1000, 9999),
            "customer_name": fullname2,
            "customer_email": email2,
            "issue": random.choice(["Cannot login", "Page not loading", "Error message"]),
            "resolution": random.choice(["Password reset", "Browser cache cleared", "Software updated"]),
        }
        
        # Generate the document text
        document = template.format(**doc_data)
        
        # Add additional PII elements randomly throughout the document
        pii_snippets = [
            f"\n\nContact phone: {phone1}",
            f"\nAlternate phone: {phone2}",
            f"\nSSN: {ssn}",
            f"\nCredit Card: {credit_card}",
            f"\nBitcoin Wallet: {bitcoin}",
            f"\nEthereum Address: {ethereum}",
            f"\nIBAN: {iban}",
            f"\nBIC: {bic}",
            f"\nIP Address: {ipv4}",
            f"\nIPv6 Address: {ipv6}",
            f"\nMAC Address: {mac}",
            f"\nWebsite: {url}",
            f"\nGPS Coordinates: {gps}",
            f"\nJob Title: {job_title}",
            f"\nDepartment: {department}",
            f"\nCounty: {random.choice(self.counties)}",
            f"\nUser Agent: {random.choice(self.user_agents)}",
            f"\nDevice: {random.choice(self.devices)}",
        ]
        
        # Add 3-5 random PII snippets to the document
        num_snippets = random.randint(3, 5)
        for snippet in random.sample(pii_snippets, num_snippets):
            insert_pos = random.randint(len(document) // 2, len(document))
            document = document[:insert_pos] + snippet + document[insert_pos:]
        
        # Add some random non-PII text to increase length
        if len(document.split()) < 100:
            document += "\n\n" + self.generate_additional_content()
        
        return {
            "id": doc_id,
            "type": doc_type,
            "text": document,
            "pii_entities": self.extract_pii_entities(document)
        }
    
    def generate_body_content(self):
        """Generate random body content for documents."""
        paragraphs = [
            "I hope this message finds you well. I wanted to follow up on our recent conversation regarding the upcoming project deadline.",
            "Please find attached the requested documents for your review. Let me know if you need any additional information.",
            "We have completed the initial analysis and would like to schedule a meeting to discuss the findings and next steps.",
            "Thank you for your prompt response. I have reviewed the information and have a few questions for clarification.",
            "Based on our discussion, I've outlined the key action items and proposed timeline for implementation.",
            "The team has made significant progress on the deliverables, and we are on track to meet the scheduled milestones.",
            "I would appreciate your feedback on the proposed approach before we proceed with the implementation phase.",
            "Please confirm receipt of this message and let me know if there are any scheduling conflicts for the proposed meeting time.",
        ]
        
        return "\n\n".join(random.sample(paragraphs, random.randint(2, 4)))
    
    def generate_additional_content(self):
        """Generate additional non-PII content to reach target length."""
        topics = [
            "The implementation of new security protocols has significantly improved our system's resilience against potential threats.",
            "Market analysis indicates a growing trend toward digital transformation across all industry sectors.",
            "Customer feedback has been overwhelmingly positive regarding the recent user interface updates.",
            "The quarterly financial report shows steady growth in key performance indicators across all regions.",
            "Technical documentation has been updated to reflect the latest changes in the software architecture.",
            "Team collaboration has been enhanced through the adoption of new communication tools and workflows.",
            "Quality assurance testing has identified several areas for improvement in the current release cycle.",
            "Strategic planning sessions have focused on identifying new opportunities for market expansion.",
        ]
        
        return "\n".join(random.sample(topics, random.randint(2, 3)))
    
    def extract_pii_entities(self, text):
        """Extract and categorize PII entities from text (simplified version)."""
        entities = []
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text, re.IGNORECASE):
            entities.append({
                "entity": "EMAIL",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # Phone patterns
        phone_pattern = r'\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            entities.append({
                "entity": "PHONE",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # SSN pattern
        ssn_pattern = r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            entities.append({
                "entity": "SSN",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # Credit card pattern (simplified)
        cc_pattern = r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b'
        for match in re.finditer(cc_pattern, text):
            entities.append({
                "entity": "CREDIT_CARD",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # IP address patterns
        ipv4_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        for match in re.finditer(ipv4_pattern, text):
            entities.append({
                "entity": "IP_ADDRESS",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # Bitcoin address pattern (simplified)
        bitcoin_pattern = r'\b1[a-km-zA-HJ-NP-Z1-9]{25,34}\b'
        for match in re.finditer(bitcoin_pattern, text):
            entities.append({
                "entity": "BITCOIN_ADDRESS",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        return entities
    
    def generate_documents(self, num_documents=50):
        """Generate multiple documents."""
        documents = []
        
        print(f"Generating {num_documents} documents with diverse PII...")
        
        for i in range(num_documents):
            doc_id = f"DOC_{i+1:03d}"
            doc = self.generate_document(doc_id)
            documents.append(doc)
            
            # Save as text file
            txt_path = self.output_dir / f"{doc_id}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(doc['text'])
            
            # Save metadata as JSON
            meta_path = self.output_dir / f"{doc_id}_meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "id": doc['id'],
                    "type": doc['type'],
                    "pii_count": len(doc['pii_entities']),
                    "pii_entities": doc['pii_entities']
                }, f, indent=2)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_documents} documents")
        
        # Save summary
        summary = {
            "total_documents": len(documents),
            "generation_date": datetime.now().isoformat(),
            "document_types": {},
            "pii_statistics": {
                "total_pii_entities": sum(len(doc['pii_entities']) for doc in documents),
                "pii_types": {}
            }
        }
        
        # Count document types
        for doc in documents:
            doc_type = doc['type']
            summary["document_types"][doc_type] = summary["document_types"].get(doc_type, 0) + 1
            
            # Count PII types
            for entity in doc['pii_entities']:
                pii_type = entity['entity']
                summary["pii_statistics"]["pii_types"][pii_type] = \
                    summary["pii_statistics"]["pii_types"].get(pii_type, 0) + 1
        
        with open(self.output_dir / "generation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Generated {num_documents} documents in '{self.output_dir}'")
        print(f"   Total PII entities: {summary['pii_statistics']['total_pii_entities']}")
        print(f"   Document types: {dict(summary['document_types'])}")
        
        return documents


def main():
    """Main function to generate documents."""
    generator = PIIDocumentGenerator("pii_documents")
    documents = generator.generate_documents(50)
    
    # Print sample document
    if documents:
        print("\n" + "="*80)
        print("SAMPLE DOCUMENT:")
        print("="*80)
        sample = documents[0]
        print(f"ID: {sample['id']}")
        print(f"Type: {sample['type']}")
        print(f"PII entities found: {len(sample['pii_entities'])}")
        print("\nText preview (first 500 chars):")
        print("-"*80)
        print(sample['text'][:500] + "..." if len(sample['text']) > 500 else sample['text'])
        print("\nPII entities detected:")
        for entity in sample['pii_entities'][:5]:  # Show first 5
            print(f"  - {entity['entity']}: {entity['value']}")
        
        print("\n" + "="*80)
        print("Files created:")
        print("="*80)
        for file in sorted(generator.output_dir.glob("*.txt")):
            print(f"  {file.name}")
        print(f"\n  generation_summary.json")
        print(f"\nTotal: {len(list(generator.output_dir.glob('*.txt')))} text files + metadata")


if __name__ == "__main__":
    main()
