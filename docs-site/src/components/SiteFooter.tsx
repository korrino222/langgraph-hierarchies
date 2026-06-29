import { Icon } from '@mintlify/components';

const socials = [
  {
    type: 'github',
    url: 'https://github.com/korrino222/langgraph-hierarchies',
  },
];

const socialIconMap: Record<string, string> = {
  github: 'github',
};

export default function SiteFooter() {
  return (
    <footer className="border-t border-gray-200 mt-24">
      <div className="flex gap-6 items-center py-10">
        {socials.map((social) => {
          const iconName =
            socialIconMap[social.type.toLowerCase()] || social.type;
          return (
            <a
              key={social.url}
              href={social.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-gray-600 transition-colors"
              aria-label={social.type}
            >
              <Icon icon={iconName} size={20} color="currentColor" />
            </a>
          );
        })}
      </div>
    </footer>
  );
}
